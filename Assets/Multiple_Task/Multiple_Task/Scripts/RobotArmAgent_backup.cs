using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class RobotArmAgent_backup : Agent
{
    public Transform target;        
    public Transform endEffector;   
    public ArticulationBody[] joints; 
    public Transform dropZone;      

    // 【新增】：顶部快门相机
    public Camera topDownCamera;    
    public float gripperLength = 0.136f;

    // 【新增】：用于存储视觉算法识别出的静态坐标
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

        // 物块生成后，强制相机渲染一帧
        if (topDownCamera != null)
        {
            topDownCamera.Render();
            ExtractTargetCoordinatesFromImage();
        }
    }

    // 【新增】：传统 CV 色彩过滤算法
    private void ExtractTargetCoordinatesFromImage()
    {
        RenderTexture rt = topDownCamera.targetTexture;
        if (rt == null)
        {
            Debug.LogWarning("RenderTexture 未设置！");
            return;
        }

        // 创建临时纹理读取相机画面
        Texture2D tex = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
        tex.Apply();

        Color[] pixels = tex.GetPixels();
        double sumX = 0;           // 使用 double 防止大量像素累加时精度丢失
        double sumY = 0;
        int targetPixelCount = 0;

        // 遍历所有像素，只检测纯蓝色（B > 0.8，R 和 G < 0.2）
        for (int i = 0; i < pixels.Length; i++)
        {
            if (pixels[i].r < 0.2f && pixels[i].g < 0.2f && pixels[i].b > 0.8f)
            {
                int x = i % rt.width;           // 横坐标（左→右）
                int y = i / rt.width;           // 纵坐标（下→上，符合 Viewport 坐标系）
                sumX += x;
                sumY += y;
                targetPixelCount++;
            }
        }

        // 释放临时纹理，防止内存泄漏
        Destroy(tex);

        if (targetPixelCount > 0)
        {
            float avgPixelX = (float)(sumX / targetPixelCount);
            float avgPixelY = (float)(sumY / targetPixelCount);

            // 计算相机到桌面的距离（用于 ViewportToWorldPoint 的深度）
            float distanceToTable = topDownCamera.transform.position.y - target.position.y;

            Vector3 viewportPos = new Vector3(
                avgPixelX / rt.width,      // 0~1
                avgPixelY / rt.height,     // 0~1（底部为 0）
                distanceToTable            // 深度
            );

            predictedTargetPos = topDownCamera.ViewportToWorldPoint(viewportPos);
            predictedTargetPos.y = target.position.y;   // 强制贴在桌面高度
        }
        else
        {
            Debug.LogWarning("视觉丢失：没找到蓝色像素！画面可能被遮挡或颜色不对。");
            predictedTargetPos = target.position; // 保底方案
        }

        // 刷新紫色 Debug 小球位置
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
        // 1. 获取各个平面的物理高度基准
        float dropZoneTopY = dropZone.GetComponent<Collider>().bounds.max.y;
        BoxCollider targetCol = target.GetComponent<BoxCollider>();
        float targetHeight = targetCol.size.y * target.transform.lossyScale.y;
        // 目标顶部高度 = 视觉预测的地面 Y 坐标 + 物块本身厚度的一半
        float targetTopY = predictedTargetPos.y + (targetHeight / 2f); 
        float flangeOffset = 0.01f + gripperLength; 

        // 2. 将纯平面的视觉坐标 predictedTargetPos，重构为带有状态机高度的 3D 坐标
        Vector3 predictedCurrentGoalPos = predictedTargetPos;

        if (currentState == 0) predictedCurrentGoalPos = new Vector3(predictedTargetPos.x, targetTopY + 0.05f + gripperLength, predictedTargetPos.z);       
        if (currentState == 1) predictedCurrentGoalPos = new Vector3(predictedTargetPos.x, targetTopY + flangeOffset, predictedTargetPos.z);      
        if (currentState == 2) predictedCurrentGoalPos = new Vector3(dropZone.position.x, dropZoneTopY + targetHeight + 0.15f + gripperLength, dropZone.position.z);     
        if (currentState == 3) predictedCurrentGoalPos = new Vector3(dropZone.position.x, dropZoneTopY + targetHeight + flangeOffset, dropZone.position.z);    
        if (currentState == 4) predictedCurrentGoalPos = new Vector3(dropZone.position.x, dropZoneTopY + targetHeight + 0.15f + gripperLength, dropZone.position.z);     

        // 3. 用重构后的坐标计算方向和距离，喂给 17 维旧模型
        Vector3 dirToPredictedGoal = predictedCurrentGoalPos - endEffector.position;
        
        sensor.AddObservation(dirToPredictedGoal.normalized); // 3维
        sensor.AddObservation(dirToPredictedGoal.magnitude);  // 1维
        sensor.AddObservation(endEffector.forward);           // 3维
        sensor.AddObservation(endEffector.up);                // 3维
        
        foreach (var joint in joints)
        {
            sensor.AddObservation(joint.jointPosition[0]);    // 6维
        }
        sensor.AddObservation((float)currentState);           // 1维
    }

    private Vector3 GetCurrentGoalPosition()
    {
        float dropZoneTopY = dropZone.GetComponent<Collider>().bounds.max.y;
        BoxCollider targetCol = target.GetComponent<BoxCollider>();
        float targetHeight = targetCol.size.y * target.transform.lossyScale.y;
        float targetTopY = target.position.y + (targetHeight / 2f); 
        float flangeOffset = 0.01f + gripperLength; 

        if (currentState == 0) return new Vector3(target.position.x, targetTopY + 0.05f + gripperLength, target.position.z);       
        if (currentState == 1) return new Vector3(target.position.x, targetTopY + flangeOffset, target.position.z);      
        if (currentState == 2) return new Vector3(dropZone.position.x, dropZoneTopY + targetHeight + 0.15f + gripperLength, dropZone.position.z);     
        if (currentState == 3) return new Vector3(dropZone.position.x, dropZoneTopY + targetHeight + flangeOffset, dropZone.position.z);    
        if (currentState == 4) return new Vector3(dropZone.position.x, dropZoneTopY + targetHeight + 0.15f + gripperLength, dropZone.position.z);     
        
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

        Vector3 offsetGoalPos = GetCurrentGoalPosition();
        float distanceToGoal = Vector3.Distance(endEffector.position, offsetGoalPos);
        float alignmentPenalty = 1.0f - Vector3.Dot(endEffector.forward, Vector3.down);
        float stepPenalty = -distanceToGoal * 0.01f - alignmentPenalty * 0.05f;

        Vector3 envLocalTargetPos = originalTargetParent.InverseTransformPoint(target.position);
        Vector2 targetXZ = new Vector2(envLocalTargetPos.x, envLocalTargetPos.z);
        if (envLocalTargetPos.y < -0.1f || targetXZ.magnitude > 0.9f) 
        {
            SetReward(-100.0f); 
            EndEpisode(); 
            return;
        }
        
        Vector2 effXZ = new Vector2(endEffector.position.x, endEffector.position.z);
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
                float xzDeviation = Vector2.Distance(effXZ, tarXZ);
                AddReward(-xzDeviation * 0.05f); 

                if (distanceToGoal < currentReachThreshold && alignmentPenalty < 0.02f && xzDeviation < currentReachThreshold)
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
                        //target.position = endEffector.position + endEffector.forward * (targetHeight / 2f + 0.01f + gripperLength);
                        StartCoroutine(SmoothGrabTransition(targetHeight));


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
                float dropXzDeviation = Vector2.Distance(effXZ, dropXZ);
                AddReward(-dropXzDeviation * 0.05f);

                if (distanceToGoal < currentReachThreshold && alignmentPenalty < 0.1f && dropXzDeviation < currentReachThreshold)
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
        float duration = 0.2f; // 过渡时间设定为 0.2 秒，可根据视觉效果微调
        Vector3 startPos = target.position;

        while (elapsedTime < duration)
        {
            // 实时获取夹爪当前的预期中心点位置
            Vector3 dynamicTargetPos = endEffector.position + endEffector.forward * (targetHeight / 2f + 0.01f + gripperLength);
            
            // 线性插值平滑移动
            target.position = Vector3.Lerp(startPos, dynamicTargetPos, elapsedTime / duration);
            elapsedTime += Time.deltaTime;
            
            yield return null; // 等待下一帧
        }
        
        // 循环结束后，确保物块彻底对齐最终位置
        target.position = endEffector.position + endEffector.forward * (targetHeight / 2f + 0.01f + gripperLength);
    }
}