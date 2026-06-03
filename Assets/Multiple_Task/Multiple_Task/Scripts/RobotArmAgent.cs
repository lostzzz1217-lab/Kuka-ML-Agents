using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class RobotArmAgent : Agent
{
    public Transform target;        
    public Transform endEffector;   
    public ArticulationBody[] joints; 
    public Transform dropZone;      

    public Camera topDownCamera;    
    
    // 【新增】：夹爪偏移量与训练模式开关
    public float gripperLength = 0.136f; 
    public bool isTrainingMode = true; // 训练时勾选，推理演示时取消勾选

    private GameObject debugSphere;
    private Vector3 predictedTargetPos; 

    private Transform originalTargetParent; 
    private Rigidbody targetRb; 
    private int currentState = 0;           
    private int pauseCounter = 0;           

    public override void Initialize()
    {
        originalTargetParent = target.parent; 
        targetRb = target.GetComponent<Rigidbody>();

        foreach (var joint in joints)
        {
            var drive = joint.xDrive;
            drive.stiffness = 100000f; 
            drive.damping = 10000f;    
            drive.forceLimit = 10000f; 
            joint.xDrive = drive;
        }
    }

    private void RespawnTarget()
    {
        Vector3 randomPos;
        float randomDistance;
        bool isInsideDropZoneColumn;

        BoxCollider targetCol = target.GetComponent<BoxCollider>();
        float localTargetHeight = targetCol.size.y * target.transform.localScale.y;

        do
        {
            randomPos = Random.insideUnitSphere;
            randomPos.y = 0f; 
            randomPos.z = Mathf.Abs(randomPos.z); 
            
            if (randomPos.magnitude < 0.001f) randomPos = Vector3.forward;

            float minReach = 0.4f;
            float maxReach = 0.55f;
            randomDistance = Random.Range(minReach, maxReach);
            
            target.localPosition = randomPos.normalized * randomDistance + new Vector3(0, localTargetHeight / 2f, 0); 
            
            Vector2 targetXZ = new Vector2(target.position.x, target.position.z);
            Vector2 dropZoneXZ = new Vector2(dropZone.position.x, dropZone.position.z);
            
            isInsideDropZoneColumn = Vector2.Distance(targetXZ, dropZoneXZ) < 0.25f;
            
        } 
        while (isInsideDropZoneColumn);

        // 【修改】：训练模式下直接传递精确坐标并加入微小噪声，跳过极其耗时的相机渲染
        if (isTrainingMode)
        {
            predictedTargetPos = target.position + new Vector3(Random.Range(-0.005f, 0.005f), 0, Random.Range(-0.005f, 0.005f));
            if (debugSphere == null)
            {
                debugSphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                debugSphere.transform.localScale = new Vector3(0.04f, 0.04f, 0.04f);
                debugSphere.GetComponent<Collider>().enabled = false;
                debugSphere.GetComponent<Renderer>().material.color = Color.magenta;
            }
            debugSphere.transform.position = predictedTargetPos;
        }
        else if (topDownCamera != null)
        {
            topDownCamera.Render();
            ExtractTargetCoordinatesFromImage();
        }
    }

    private void ExtractTargetCoordinatesFromImage()
    {
        RenderTexture rt = topDownCamera.targetTexture;
        if (rt == null) return;

        Texture2D tex = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
        tex.Apply();

        Color[] pixels = tex.GetPixels();
        double sumX = 0;           
        double sumY = 0;
        int targetPixelCount = 0;

        for (int i = 0; i < pixels.Length; i++)
        {
            if (pixels[i].r < 0.2f && pixels[i].g < 0.2f && pixels[i].b > 0.8f)
            {
                int x = i % rt.width;           
                int y = i / rt.width;           
                sumX += x;
                sumY += y;
                targetPixelCount++;
            }
        }

        Destroy(tex);

        if (targetPixelCount > 0)
        {
            float avgPixelX = (float)(sumX / targetPixelCount);
            float avgPixelY = (float)(sumY / targetPixelCount);

            float distanceToTable = topDownCamera.transform.position.y - target.position.y;

            Vector3 viewportPos = new Vector3(
                avgPixelX / rt.width,      
                avgPixelY / rt.height,     
                distanceToTable            
            );

            predictedTargetPos = topDownCamera.ViewportToWorldPoint(viewportPos);
            predictedTargetPos.y = target.position.y;   
        }
        else
        {
            predictedTargetPos = target.position; 
        }

        if (debugSphere == null)
        {
            debugSphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            debugSphere.transform.localScale = new Vector3(0.04f, 0.04f, 0.04f);
            debugSphere.GetComponent<Collider>().enabled = false;
            debugSphere.GetComponent<Renderer>().material.color = Color.magenta;
        }
        debugSphere.transform.position = predictedTargetPos;
    }

    public override void OnEpisodeBegin()
    {
        currentState = 0;
        pauseCounter = 0;
        target.SetParent(originalTargetParent);
        target.GetComponent<Collider>().enabled = true;

        if (targetRb != null) 
        {
            targetRb.isKinematic = false; 
            targetRb.constraints = RigidbodyConstraints.FreezePositionX | RigidbodyConstraints.FreezePositionZ | RigidbodyConstraints.FreezeRotation;
            targetRb.linearVelocity = Vector3.zero;
            targetRb.angularVelocity = Vector3.zero;
        }

        RespawnTarget();

        float[] homeAngles = new float[6] { 0f, -90f, 90f, 0f, 0f, 0f };
        for (int i = 0; i < joints.Length; i++)
        {
            joints[i].SetDriveTarget(ArticulationDriveAxis.X, homeAngles[i]);
            joints[i].jointPosition = new ArticulationReducedSpace(homeAngles[i] * Mathf.Deg2Rad);
            joints[i].jointVelocity = new ArticulationReducedSpace(0f); 
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        float dropZoneTopY = dropZone.GetComponent<Collider>().bounds.max.y;
        BoxCollider targetCol = target.GetComponent<BoxCollider>();
        float targetHeight = targetCol.size.y * target.transform.lossyScale.y;
        float targetTopY = predictedTargetPos.y + (targetHeight / 2f); 
        float flangeOffset = 0.01f; 

        // 目标依然是这个位置，但待会儿去追踪这个位置的执行器变成了 TCP (夹爪尖端)
        Vector3 predictedCurrentGoalPos = predictedTargetPos;

        if (currentState == 0) predictedCurrentGoalPos = new Vector3(predictedTargetPos.x, targetTopY + 0.05f, predictedTargetPos.z);       
        if (currentState == 1) predictedCurrentGoalPos = new Vector3(predictedTargetPos.x, targetTopY + flangeOffset, predictedTargetPos.z);      
        if (currentState == 2) predictedCurrentGoalPos = new Vector3(dropZone.position.x, dropZoneTopY + targetHeight + 0.15f, dropZone.position.z);     
        if (currentState == 3) predictedCurrentGoalPos = new Vector3(dropZone.position.x, dropZoneTopY + targetHeight + flangeOffset, dropZone.position.z);    
        if (currentState == 4) predictedCurrentGoalPos = new Vector3(dropZone.position.x, dropZoneTopY + targetHeight + 0.15f, dropZone.position.z);     

        // 【修改】：计算基于夹爪尖端的向量
        Vector3 currentTCP = endEffector.position + endEffector.forward * gripperLength;
        Vector3 dirToPredictedGoal = predictedCurrentGoalPos - currentTCP;
        
        sensor.AddObservation(dirToPredictedGoal.normalized); 
        sensor.AddObservation(dirToPredictedGoal.magnitude);  
        sensor.AddObservation(endEffector.forward);           
        sensor.AddObservation(endEffector.up);                
        
        foreach (var joint in joints)
        {
            sensor.AddObservation(joint.jointPosition[0]);    
        }
        sensor.AddObservation((float)currentState);           
    }

    private Vector3 GetCurrentGoalPosition()
    {
        float dropZoneTopY = dropZone.GetComponent<Collider>().bounds.max.y;
        BoxCollider targetCol = target.GetComponent<BoxCollider>();
        float targetHeight = targetCol.size.y * target.transform.lossyScale.y;
        float targetTopY = target.position.y + (targetHeight / 2f); 
        float flangeOffset = 0.01f; 

        if (currentState == 0) return new Vector3(target.position.x, targetTopY + 0.05f, target.position.z);       
        if (currentState == 1) return new Vector3(target.position.x, targetTopY + flangeOffset, target.position.z);      
        if (currentState == 2) return new Vector3(dropZone.position.x, dropZoneTopY + targetHeight + 0.15f, dropZone.position.z);     
        if (currentState == 3) return new Vector3(dropZone.position.x, dropZoneTopY + targetHeight + flangeOffset, dropZone.position.z);    
        if (currentState == 4) return new Vector3(dropZone.position.x, dropZoneTopY + targetHeight + 0.15f, dropZone.position.z);     
        
        return target.position;
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        if (pauseCounter == 0) 
        {
            float actionMagnitude = 0f;
            for (int i = 0; i < 6; i++)
            {
                float actionVal = Mathf.Clamp(actionBuffers.ContinuousActions[i], -1f, 1f);
                actionMagnitude += Mathf.Abs(actionVal);
                float deltaAngle = actionVal * 2.0f; 
                float currentTarget = joints[i].xDrive.target;
                joints[i].SetDriveTarget(ArticulationDriveAxis.X, currentTarget + deltaAngle);
            }
            AddReward(-actionMagnitude * 0.0001f);
        }

        float currentReachThreshold = Academy.Instance.EnvironmentParameters.GetWithDefault("reach_threshold", 0.05f);
        float currentAlignmentThreshold = Academy.Instance.EnvironmentParameters.GetWithDefault("alignment_threshold", 0.02f);

        Vector3 offsetGoalPos = GetCurrentGoalPosition();
        
        // 【修改】：所有距离计算全部基于当前 TCP 的位置
        Vector3 currentTCP = endEffector.position + endEffector.forward * gripperLength;
        float distanceToGoal = Vector3.Distance(currentTCP, offsetGoalPos);
        float alignmentPenalty = 1.0f - Vector3.Dot(endEffector.forward, Vector3.down);
        
        // 【修改】：大幅提升姿态惩罚权重
        float stepPenalty = -distanceToGoal * 0.01f - alignmentPenalty * 0.05f;

        // 【修改】：增加夹爪直接撞击桌面的“死亡惩罚”
        if (currentTCP.y < 0.02f) 
        {
            SetReward(-50.0f); 
            EndEpisode();
            return;
        }

        Vector3 envLocalTargetPos = originalTargetParent.InverseTransformPoint(target.position);
        Vector2 targetXZ = new Vector2(envLocalTargetPos.x, envLocalTargetPos.z);
        if (envLocalTargetPos.y < -0.1f || targetXZ.magnitude > 0.9f) 
        {
            SetReward(-100.0f); 
            EndEpisode(); 
            return;
        }
        
        // 【修改】：基于 TCP 进行水平偏差计算
        Vector2 tcpXZ = new Vector2(currentTCP.x, currentTCP.z);
        
        switch (currentState)
        {
            case 0: 
                if (distanceToGoal < currentReachThreshold * 1.5f && alignmentPenalty < 0.1f)
                {   
                    pauseCounter++;
                    if(pauseCounter > 25)
                    {
                        currentState = 1;
                        pauseCounter = 0;
                    }
                }
                else
                {
                    pauseCounter = 0;
                    AddReward(stepPenalty);
                }
                break;

            case 1: 
                Vector2 tarXZ = new Vector2(target.position.x, target.position.z);
                float xzDeviation = Vector2.Distance(tcpXZ, tarXZ);
                AddReward(-xzDeviation * 0.05f); 

                // 【修改】：收紧对齐判定阈值，从 0.1f 收紧至 0.02f
                if (distanceToGoal < currentReachThreshold && alignmentPenalty < currentAlignmentThreshold && xzDeviation < currentReachThreshold)
                {
                    pauseCounter++;
                    AddReward(0.01f);

                    float requiredGrabFrames = Academy.Instance.EnvironmentParameters.GetWithDefault("grab_pause_frames", 50.0f);
                    if (pauseCounter >= requiredGrabFrames)
                    {
                        currentState = 2;
                        if (targetRb != null) targetRb.isKinematic = true;
                        target.GetComponent<Collider>().enabled = false;
                        target.SetParent(endEffector);
                        
                        BoxCollider targetCol = target.GetComponent<BoxCollider>();
                        float targetHeight = targetCol.size.y * target.transform.lossyScale.y;
                        
                        // 【修改】：启动平滑吸附过渡协程
                        if (!isTrainingMode) {
                            StartCoroutine(SmoothGrabTransition(targetHeight));
                        } else {
                            target.position = endEffector.position + endEffector.forward * (targetHeight / 2f + 0.01f + gripperLength);
                        }
                        
                        pauseCounter = 0;
                        AddReward(30.0f);
                    }
                }
                else
                {
                    pauseCounter = Mathf.Max(0, pauseCounter - 2); 
                    AddReward(stepPenalty); 
                }
                break;

            case 2: 
                if (distanceToGoal < currentReachThreshold * 1.5f && alignmentPenalty < 0.1f)
                {
                    pauseCounter++;
                    if(pauseCounter > 25)
                    {
                        currentState = 3;
                        pauseCounter = 0;
                    }
                }
                else
                {
                    pauseCounter = 0;
                    AddReward(stepPenalty);
                }
                break;

            case 3: 
                Vector2 dropXZ = new Vector2(dropZone.position.x, dropZone.position.z);
                float dropXzDeviation = Vector2.Distance(tcpXZ, dropXZ);
                AddReward(-dropXzDeviation * 0.05f);

                if (distanceToGoal < currentReachThreshold && alignmentPenalty < currentAlignmentThreshold && dropXzDeviation < currentReachThreshold)
                {
                    pauseCounter++;
                    AddReward(0.01f);

                    float requiredDropFrames = Academy.Instance.EnvironmentParameters.GetWithDefault("drop_pause_frames", 50.0f);
                    if (pauseCounter >= requiredDropFrames)
                    {
                        AddReward(100.0f);
                        target.SetParent(originalTargetParent);
                        if (targetRb != null)
                        {
                            targetRb.isKinematic = false;
                            targetRb.constraints = RigidbodyConstraints.None;
                            targetRb.linearVelocity = Vector3.zero;
                            targetRb.angularVelocity = Vector3.zero;
                        }
                        target.GetComponent<Collider>().enabled = true;
                        currentState = 4; 
                        pauseCounter = 0;
                    }
                }
                else
                {
                    pauseCounter = Mathf.Max(0, pauseCounter - 2);
                    AddReward(stepPenalty);
                }
                break;

            case 4: 
                if (distanceToGoal < 0.08f) 
                {
                    AddReward(15.0f); 
                    if (targetRb != null) 
                    {
                        targetRb.isKinematic = true; 
                        targetRb.linearVelocity = Vector3.zero;
                        targetRb.angularVelocity = Vector3.zero;
                    }
                    
                    RespawnTarget(); 

                    if (targetRb != null)
                    {
                        targetRb.isKinematic = false; 
                        targetRb.constraints = RigidbodyConstraints.FreezePositionX | RigidbodyConstraints.FreezePositionZ | RigidbodyConstraints.FreezeRotation;
                    }
                    
                    currentState = 0; 
                }
                else
                {
                    AddReward(stepPenalty);
                }
                break;
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxis("Horizontal"); 
        continuousActionsOut[1] = Input.GetAxis("Vertical");   
        for (int i = 2; i < 6; i++) continuousActionsOut[i] = 0f;
    }

    // 【新增】：平滑吸附过渡协程
    private System.Collections.IEnumerator SmoothGrabTransition(float targetHeight)
    {
        float elapsedTime = 0f;
        float duration = 0.2f; 
        Vector3 startPos = target.position;

        while (elapsedTime < duration)
        {
            Vector3 dynamicTargetPos = endEffector.position + endEffector.forward * (targetHeight / 2f + 0.01f + gripperLength);
            target.position = Vector3.Lerp(startPos, dynamicTargetPos, elapsedTime / duration);
            elapsedTime += Time.deltaTime;
            yield return null; 
        }
        
        target.position = endEffector.position + endEffector.forward * (targetHeight / 2f + 0.01f + gripperLength);
    }
}