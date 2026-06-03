using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class RobotArmAgent_Vector : Agent
{
    public Transform target;        
    public Transform endEffector;   
    public ArticulationBody[] joints; 
    public Transform dropZone;      

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
        float targetHeight = targetCol.size.y * target.transform.lossyScale.y;

        do
        {
            randomPos = Random.insideUnitSphere;
            randomPos.y = 0f; 
            randomPos.z = Mathf.Abs(randomPos.z); 
            
            if (randomPos.magnitude < 0.001f) randomPos = Vector3.forward;

            float minReach = 0.4f;
            float maxReach = 0.55f;
            randomDistance = Random.Range(minReach, maxReach);
            
            target.localPosition = randomPos.normalized * randomDistance + new Vector3(0, targetHeight / 2f, 0); 
            
            Vector2 targetXZ = new Vector2(target.position.x, target.position.z);
            Vector2 dropZoneXZ = new Vector2(dropZone.position.x, dropZone.position.z);
            
            isInsideDropZoneColumn = Vector2.Distance(targetXZ, dropZoneXZ) < 0.25f;
            
        } 
        while (isInsideDropZoneColumn);
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
            // 物理修复：在桌面生成时冻结水平位移，防止未抓取前被碰飞
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
        // 恢复模型所需的 17 维核心数据输入
        Vector3 currentGoalPos = GetCurrentGoalPosition();
        Vector3 dirToGoal = currentGoalPos - endEffector.position;
        
        sensor.AddObservation(dirToGoal.normalized); // 3
        sensor.AddObservation(dirToGoal.magnitude);  // 1
        sensor.AddObservation(endEffector.forward);  // 3
        sensor.AddObservation(endEffector.up);       // 3
        
        foreach (var joint in joints)
        {
            sensor.AddObservation(joint.jointPosition[0]); // 6
        }
        sensor.AddObservation((float)currentState);  // 1
        // 总计 = 17
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

        // 硬编码回旧模型的严苛阈值
        float strictThreshold = 0.02f;

        Vector3 offsetGoalPos = GetCurrentGoalPosition();
        float distanceToGoal = Vector3.Distance(endEffector.position, offsetGoalPos);
        float alignmentPenalty = 1.0f - Vector3.Dot(endEffector.forward, Vector3.down);
        float stepPenalty = -distanceToGoal * 0.01f - alignmentPenalty * 0.005f;

        // 物理修复：环境绝对坐标的越界保护
        Vector3 envLocalTargetPos = originalTargetParent.InverseTransformPoint(target.position);
        Vector2 targetXZ = new Vector2(envLocalTargetPos.x, envLocalTargetPos.z);
        if (envLocalTargetPos.y < -0.1f || targetXZ.magnitude > 0.65f) 
        {
            SetReward(-100.0f); 
            EndEpisode(); 
            return;
        }
        
        Vector2 effXZ = new Vector2(endEffector.position.x, endEffector.position.z);
        switch (currentState)
        {
            case 0: 
                if (distanceToGoal < strictThreshold * 1.5f && alignmentPenalty < 0.1f)
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

                if (distanceToGoal < strictThreshold && alignmentPenalty < 0.1f && xzDeviation < strictThreshold)
                {
                    pauseCounter++;
                    AddReward(0.01f);

                    if (pauseCounter >= 200) // 硬编码抓取等待帧
                    {
                        currentState = 2;
                        if (targetRb != null) targetRb.isKinematic = true;
                        target.GetComponent<Collider>().enabled = false;
                        target.SetParent(endEffector);
                        
                        // 量子吸附：让视觉表现极其稳定
                        BoxCollider targetCol = target.GetComponent<BoxCollider>();
                        float targetHeight = targetCol.size.y * target.transform.lossyScale.y;
                        target.position = endEffector.position + endEffector.forward * (targetHeight / 2f + 0.01f);
                        
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
                if (distanceToGoal < strictThreshold * 1.5f && alignmentPenalty < 0.1f)
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

                if (distanceToGoal < strictThreshold && alignmentPenalty < 0.1f && dropXzDeviation < strictThreshold)
                {
                    pauseCounter++;
                    AddReward(0.01f);

                    if (pauseCounter >= 200) // 硬编码放置等待帧
                    {
                        AddReward(100.0f);
                        target.SetParent(originalTargetParent);
                        if (targetRb != null)
                        {
                            targetRb.isKinematic = false;
                            // 物理修复：解冻所有约束，允许物块在放置区发生物理自然碰撞
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
}