using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;
using UnityEngine.Splines;
using Unity.Mathematics;

public class CamWelding : Agent
{
    public SplineContainer seamSpline;
    public Transform endEffector; 
    public ArticulationBody[] robotJoints; 
    public float standOffDistance = 0.03f;
    
    private float3[] originalKnots;
    private Vector3 originalSeamLocalPos; 
    
    private float currentProgressT = 0f; 
    private float maxDistanceAllowed = 3.0f; 

    // === 悬停状态机变量 ===
    private bool isHoverComplete = false; 
    private int hoverStepCount = 0;       
    public int requiredHoverSteps = 100;  

    private void ResetRobotJoints()
    {
        if (robotJoints == null) return;

        foreach (var joint in robotJoints)
        {
            joint.jointPosition = new ArticulationReducedSpace(0f);
            joint.jointVelocity = new ArticulationReducedSpace(0f);
            joint.linearVelocity = Vector3.zero;
            joint.angularVelocity = Vector3.zero;

            ArticulationDrive drive = joint.xDrive;
            drive.target = 0f; 
            joint.xDrive = drive;
        }
    }

    private void RefreshSeamLocation()
    {
        seamSpline.transform.localPosition = originalSeamLocalPos + new Vector3(
            UnityEngine.Random.Range(-0.1f, 0.1f),
            0,
            UnityEngine.Random.Range(0.10f, 0.17f) 
        );

        Spline spline = seamSpline.Spline;
        for (int i = 0; i < spline.Count; i++)
        {
            BezierKnot knot = spline[i];
            knot.Position = new float3(
                originalKnots[i].x + UnityEngine.Random.Range(-0.05f, 0.05f), 
                originalKnots[i].y, 
                originalKnots[i].z + UnityEngine.Random.Range(-0.05f, 0.05f)
            );
            spline.SetKnot(i, knot);
        }

        isHoverComplete = false;
        hoverStepCount = 0;
        currentProgressT = 0f;
    }

    public override void Initialize()
    {
        Spline spline = seamSpline.Spline;
        if (spline == null || spline.Count < 2)
        {
            this.gameObject.SetActive(false); 
            return;
        }

        originalSeamLocalPos = seamSpline.transform.localPosition;
        originalKnots = new float3[spline.Count];
        for (int i = 0; i < spline.Count; i++)
        {
            originalKnots[i] = spline[i].Position;
        }

        if (robotJoints != null)
        {
            foreach (var joint in robotJoints)
            {
                var drive = joint.xDrive;
                drive.stiffness = 100000f; 
                drive.damping = 10000f;    
                drive.forceLimit = 10000f; 
                joint.xDrive = drive;
            }
        }
    }

    public override void OnEpisodeBegin()
    {
        ResetRobotJoints();
        RefreshSeamLocation();
    }

    // ==========================================
    // 【核心重构：做减法的大脑输入】
    // ==========================================
    public override void CollectObservations(VectorSensor sensor)
    {
        // 1. 仅保留极其基础的本体感觉：6个关节的角度和速度 (6 * 2 = 12 维)
        foreach (var joint in robotJoints)
        {
            float pos = float.IsNaN(joint.jointPosition[0]) || float.IsInfinity(joint.jointPosition[0]) ? 0f : joint.jointPosition[0];
            float vel = float.IsNaN(joint.jointVelocity[0]) || float.IsInfinity(joint.jointVelocity[0]) ? 0f : joint.jointVelocity[0];
            sensor.AddObservation(pos);
            sensor.AddObservation(vel);
        }

        // 2. 当前的状态机阶段：0代表正在找起点，1代表正在焊接 (1 维)
        sensor.AddObservation(isHoverComplete ? 1.0f : 0.0f); 

        // 【注意】总 Space Size 必须在 Unity 面板中改为 13！
        // 十字误差、向量、目标坐标等参数已全部删除，Camera Sensor 会自动将画面像素输入给神经网络。
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // 1. 关节平滑驱动 
        var continuousActions = actionBuffers.ContinuousActions;
        for (int i = 0; i < robotJoints.Length; i++)
        {
            ArticulationDrive drive = robotJoints[i].xDrive;
            float actionValue = continuousActions[i];
            float deltaAngle = actionValue * 0.5f; 
            float newTarget = drive.target + deltaAngle;
            newTarget = Mathf.Clamp(newTarget, drive.lowerLimit, drive.upperLimit);
            drive.target = newTarget;
            robotJoints[i].xDrive = drive;
        }

        // 2. 后台“裁判”系统：只算距离和进度用来发钱/扣钱，绝不告诉 AI 具体数字
        float3 localEndPos = seamSpline.transform.InverseTransformPoint(endEffector.position);
        float3 localNearestPoint;
        float t;
        SplineUtility.GetNearestPoint(seamSpline.Spline, localEndPos, out localNearestPoint, out t, 4, 2);

        Vector3 worldNearestPoint = seamSpline.transform.TransformPoint(localNearestPoint);
        Vector3 targetPos;
        float distanceError;
        float alignment;

        // ==========================================
        // 【核心状态机保持不变，用作后台评价标准】
        // ==========================================
        if (!isHoverComplete)
        {
            // --- 阶段一：寻找起点并悬停 ---
            float3 localStartPos = SplineUtility.EvaluatePosition(seamSpline.Spline, 0f);
            Vector3 worldStartPos = seamSpline.transform.TransformPoint(localStartPos);
            targetPos = worldStartPos + Vector3.up * standOffDistance;
            
            distanceError = Vector3.Distance(endEffector.position, targetPos);
            Vector3 dirToSeam = (worldStartPos - endEffector.position).normalized;
            alignment = Vector3.Dot(endEffector.forward, dirToSeam);

            if (distanceError > maxDistanceAllowed)
            {
                AddReward(-50.0f);
                EndEpisode();
                return;
            }

            if (distanceError > 0.1f) {
                AddReward((1.0f - distanceError / maxDistanceAllowed - 1.0f) * 0.02f);
                hoverStepCount = 0; 
            } else {
                AddReward((1.0f - distanceError / 0.1f) * 0.005f);

                if (distanceError <= 0.05f && alignment > 0.8f) {
                    hoverStepCount++;
                    AddReward(0.01f); 
                    
                    if (hoverStepCount >= requiredHoverSteps) {
                        isHoverComplete = true; 
                        currentProgressT = 0f;
                    }
                } else {
                    hoverStepCount = 0; 
                }
            }

            if (alignment < 0.8f) AddReward((alignment - 0.8f) * 0.005f); 
            else AddReward(alignment * 0.002f);
            AddReward(-0.001f);
        }
        else
        {
            // --- 阶段二：匀速跟进焊接 ---
            targetPos = worldNearestPoint + Vector3.up * standOffDistance;
            distanceError = Vector3.Distance(endEffector.position, targetPos);
            Vector3 dirToSeam = (worldNearestPoint - endEffector.position).normalized;
            alignment = Vector3.Dot(endEffector.forward, dirToSeam);

            if (distanceError > maxDistanceAllowed) {
                AddReward(-50.0f);
                EndEpisode();
                return;
            }

            float acceptableWeldDistance = 0.10f;
            if (distanceError > acceptableWeldDistance) {
                AddReward((1.0f - distanceError / maxDistanceAllowed - 1.0f) * 0.02f); 
            } else {
                AddReward((1.0f - distanceError / acceptableWeldDistance) * 0.005f);
            }

            if (alignment < 0.8f) AddReward((alignment - 0.8f) * 0.005f); 
            else AddReward(alignment * 0.002f);
            AddReward(-0.001f);

            if (t > currentProgressT)
            {
                float progressDelta = t - currentProgressT;
                if (distanceError <= acceptableWeldDistance && alignment > 0.8f)
                {
                    float closeFactor = 1.0f - (distanceError / acceptableWeldDistance);
                    AddReward(progressDelta * 150.0f * closeFactor * alignment); 
                }
                
                if (distanceError < 0.2f) {
                    currentProgressT = t;
                }
            }

            if (currentProgressT >= 0.95f)
            {
                if (distanceError <= 0.05f) {
                    AddReward(20.0f); 
                    RefreshSeamLocation(); 
                } else {
                    AddReward(-5.0f); 
                    EndEpisode();
                }
            }
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxis("Horizontal"); 
        continuousActionsOut[1] = Input.GetAxis("Vertical");   
        for(int i = 2; i < 6; i++) {
            continuousActionsOut[i] = 0f;
        }
    }
}