using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;
using UnityEngine.Splines;
using Unity.Mathematics;

public class WeldingAgent : Agent
{
    public SplineContainer seamSpline;
    public Transform endEffector; 
    public ArticulationBody[] robotJoints; 
    public float standOffDistance = 0.03f;
    
    private float3[] originalKnots;
    private Vector3 originalSeamLocalPos; 
    
    private float currentProgressT = 0f; 
    private float maxDistanceAllowed = 3.0f; 

    // === 【新增：悬停状态机变量】 ===
    private bool isHoverComplete = false; // 是否已经完成了2秒悬停
    private int hoverStepCount = 0;       // 悬停帧数计数器
    // Unity的ML-Agents默认每秒做50次决策(FixedUpdate)，所以100步刚好是2秒
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

        // 【关键】刷新线条时，重置状态机！
        isHoverComplete = false;
        hoverStepCount = 0;
        currentProgressT = 0f;
    }

    private void AddSafeVectorObs(VectorSensor sensor, Vector3 vec, string debugName)
    {
        if (float.IsNaN(vec.x) || float.IsInfinity(vec.x) ||
            float.IsNaN(vec.y) || float.IsInfinity(vec.y) ||
            float.IsNaN(vec.z) || float.IsInfinity(vec.z))
        {
            sensor.AddObservation(Vector3.zero);
        }
        else
        {
            sensor.AddObservation(vec);
        }
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

    public override void CollectObservations(VectorSensor sensor)
    {
        foreach (var joint in robotJoints)
        {
            float pos = float.IsNaN(joint.jointPosition[0]) || float.IsInfinity(joint.jointPosition[0]) ? 0f : joint.jointPosition[0];
            float vel = float.IsNaN(joint.jointVelocity[0]) || float.IsInfinity(joint.jointVelocity[0]) ? 0f : joint.jointVelocity[0];
            sensor.AddObservation(pos);
            sensor.AddObservation(vel);
        }

        float3 localEndPos = seamSpline.transform.InverseTransformPoint(endEffector.position);
        AddSafeVectorObs(sensor, localEndPos, "localEndPos");

        // === 动态计算当前阶段的“靶心” ===
        Vector3 targetPos;
        Vector3 worldNearestPoint;
        float3 localTangent;

        if (!isHoverComplete)
        {
            // 没悬停完，靶心死死锁定在起点
            float3 localStartPos = SplineUtility.EvaluatePosition(seamSpline.Spline, 0f);
            worldNearestPoint = seamSpline.transform.TransformPoint(localStartPos);
            targetPos = worldNearestPoint + Vector3.up * standOffDistance;
            localTangent = SplineUtility.EvaluateTangent(seamSpline.Spline, 0f);
        }
        else
        {
            // 悬停完了，靶心变成随进度移动的最近点
            float3 localNearestPoint;
            float t;
            SplineUtility.GetNearestPoint(seamSpline.Spline, localEndPos, out localNearestPoint, out t, 4, 2);
            worldNearestPoint = seamSpline.transform.TransformPoint(localNearestPoint);
            targetPos = worldNearestPoint + Vector3.up * standOffDistance;
            localTangent = SplineUtility.EvaluateTangent(seamSpline.Spline, currentProgressT);
        }

        // 添加观察向量
        Vector3 errorVector = targetPos - endEffector.position;
        AddSafeVectorObs(sensor, errorVector, "errorVector");

        Vector3 worldTangent = seamSpline.transform.TransformDirection(localTangent).normalized;
        AddSafeVectorObs(sensor, worldTangent, "worldTangent");

        float lookAheadT = Mathf.Clamp01(currentProgressT + 0.1f);
        float3 localLookAheadPos = SplineUtility.EvaluatePosition(seamSpline.Spline, lookAheadT);
        Vector3 worldLookAheadPos = seamSpline.transform.TransformPoint(localLookAheadPos);
        Vector3 lookAheadDir = worldLookAheadPos - endEffector.position;
        AddSafeVectorObs(sensor, lookAheadDir, "lookAheadDir");

        AddSafeVectorObs(sensor, endEffector.forward, "endEffector.forward");
        AddSafeVectorObs(sensor, endEffector.up, "endEffector.up");

        // 【极其关键：Space Size + 1 的原因】告诉神经网络现在是处于什么阶段
        sensor.AddObservation(isHoverComplete ? 1.0f : 0.0f); 
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // 1. 关节驱动 (保持你设定的平滑速度)
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

        // 2. 状态获取
        float3 localEndPos = seamSpline.transform.InverseTransformPoint(endEffector.position);
        float3 localNearestPoint;
        float t;
        SplineUtility.GetNearestPoint(seamSpline.Spline, localEndPos, out localNearestPoint, out t, 4, 2);

        Vector3 worldNearestPoint = seamSpline.transform.TransformPoint(localNearestPoint);
        Vector3 targetPos;
        float distanceError;
        float alignment;

        // ==========================================
        // 【核心状态机：阶段分发】
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

            // 严厉的距离引导，逼迫它去起点
            if (distanceError > 0.1f) {
                AddReward((1.0f - distanceError / maxDistanceAllowed - 1.0f) * 0.02f);
                hoverStepCount = 0; // 偏离就清零计时
            } else {
                AddReward((1.0f - distanceError / 0.1f) * 0.005f);

                // 悬停质检：距离 < 5cm 且 姿态极正
                if (distanceError <= 0.05f && alignment > 0.8f) {
                    hoverStepCount++;
                    AddReward(0.01f); // 发放悬停工资，鼓励稳住
                    
                    if (hoverStepCount >= requiredHoverSteps) {
                        isHoverComplete = true; // ✨ 解锁第二阶段！
                        currentProgressT = 0f;
                    }
                } else {
                    hoverStepCount = 0; // 手抖了，重新计秒！
                }
            }

            // 姿态与时间惩罚
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

            // 必须严格贴合，否则扣分
            float acceptableWeldDistance = 0.10f;
            if (distanceError > acceptableWeldDistance) {
                AddReward((1.0f - distanceError / maxDistanceAllowed - 1.0f) * 0.02f); 
            } else {
                AddReward((1.0f - distanceError / acceptableWeldDistance) * 0.005f);
            }

            if (alignment < 0.8f) AddReward((alignment - 0.8f) * 0.005f); 
            else AddReward(alignment * 0.002f);
            AddReward(-0.001f);

            // 发放进度大奖
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

            // 终点质检
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