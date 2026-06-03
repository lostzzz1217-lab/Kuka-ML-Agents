using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;
using UnityEngine.Splines;
using Unity.Mathematics;

public class WeldingAgent : Agent
{
    // ===================== Inspector 引用 =====================

    [Header("场景引用")]
    public SplineContainer seamSpline;
    public Transform endEffector;
    public ArticulationBody[] robotJoints;

    [Header("视觉训练引用")]
    [Tooltip("用于奖励中判断焊缝目标点是否在相机视野内。真正的视觉输入请通过 Camera Sensor Component 添加。")]
    public Camera weldCamera;

    [Header("焊接参数")]
    [Tooltip("末端执行器与焊缝之间保持的距离")]
    public float standOffDistance = 0.03f;

    [Tooltip("动作对关节 target 的影响幅度")]
    public float actionScale = 0.3f;

    [Header("课程参数：可由 YAML 覆盖")]
    [Tooltip("焊缝随机扰动范围，训练初期建议为 0")]
    public float seamRandomRange = 0.0f;

    [Tooltip("允许的焊缝横向误差")]
    public float crossTrackTolerance = 0.08f;

    [Header("视觉训练辅助参数")]
    [Tooltip("接近焊缝起点时允许的距离")]
    public float startCaptureTolerance = 0.12f;

    [Tooltip("鼓励焊缝目标点位于相机画面中心的奖励权重")]
    public float cameraCenterRewardWeight = 0.002f;

    [Tooltip("惩罚过大关节动作，减少机械臂抖动")]
    public float jointMotionPenalty = 0.0001f;

    // ===================== 内部状态 =====================

    private float3[] originalKnots;
    private Vector3 originalSeamLocalPos;

    private float currentProgressT = 0f;
    private int stagnationCounter = 0;
    private int numJoints;

    private WeldingPhase phase = WeldingPhase.ApproachStart;

    // ===================== 常量约束 =====================

    private const float MAX_DISTANCE = 1.0f;
    private const int STAGNATION_LIMIT = 300;
    private const float PROGRESS_COMPLETE = 0.95f;
    private const float MIN_DOWNWARD_ALIGNMENT = 0.5f;

    private enum WeldingPhase
    {
        ApproachStart,
        TrackSeam,
        Success,
        Failed
    }

    // ===================== 初始化 =====================

    public override void Initialize()
    {
        numJoints = robotJoints != null ? robotJoints.Length : 0;

        if (numJoints == 0)
        {
            Debug.LogError("[WeldingAgent] Robot Joints 没有设置。请拖入 link_1 到 link_6。");
            gameObject.SetActive(false);
            return;
        }

        if (seamSpline == null || seamSpline.Spline == null || seamSpline.Spline.Count < 2)
        {
            Debug.LogError("[WeldingAgent] Seam Spline 没有设置，或 Spline 点数不足。");
            gameObject.SetActive(false);
            return;
        }

        if (endEffector == null)
        {
            Debug.LogError("[WeldingAgent] End Effector 没有设置。请拖入 flange。");
            gameObject.SetActive(false);
            return;
        }

        originalSeamLocalPos = seamSpline.transform.localPosition;

        Spline spline = seamSpline.Spline;
        originalKnots = new float3[spline.Count];

        for (int i = 0; i < spline.Count; i++)
        {
            originalKnots[i] = spline[i].Position;
        }

        SetupJointDrives();
    }

    private void SetupJointDrives()
    {
        foreach (var joint in robotJoints)
        {
            if (joint == null) continue;

            var drive = joint.xDrive;
            drive.stiffness = 100000f;
            drive.damping = 10000f;
            drive.forceLimit = 10000f;
            joint.xDrive = drive;
        }
    }

    // ===================== Episode 重置 =====================

    public override void OnEpisodeBegin()
    {
        var envParams = Academy.Instance.EnvironmentParameters;

        seamRandomRange = envParams.GetWithDefault("seam_random_range", seamRandomRange);
        crossTrackTolerance = envParams.GetWithDefault("cross_track_tolerance", crossTrackTolerance);

        ResetRobotJoints();
        RandomizeSeam();

        currentProgressT = 0f;
        stagnationCounter = 0;
        phase = WeldingPhase.ApproachStart;
    }

    private void ResetRobotJoints()
    {
        foreach (var joint in robotJoints)
        {
            if (joint == null) continue;

            joint.jointPosition = new ArticulationReducedSpace(0f);
            joint.jointVelocity = new ArticulationReducedSpace(0f);
            joint.linearVelocity = Vector3.zero;
            joint.angularVelocity = Vector3.zero;

            var drive = joint.xDrive;
            drive.target = 0f;
            joint.xDrive = drive;
        }
    }

    private void RandomizeSeam()
    {
        float r = seamRandomRange;

        seamSpline.transform.localPosition = originalSeamLocalPos + new Vector3(
            UnityEngine.Random.Range(-r, r),
            0f,
            UnityEngine.Random.Range(-r, r)
        );

        Spline spline = seamSpline.Spline;
        float knotRange = r * 0.5f;

        for (int i = 0; i < spline.Count; i++)
        {
            BezierKnot knot = spline[i];

            knot.Position = new float3(
                originalKnots[i].x + UnityEngine.Random.Range(-knotRange, knotRange),
                originalKnots[i].y,
                originalKnots[i].z + UnityEngine.Random.Range(-knotRange, knotRange)
            );

            spline.SetKnot(i, knot);
        }
    }

    // ==========================================================
    // 观测空间说明：
    //
    // 本版本使用：视觉输入 + 关节本体感知
    //
    // 视觉输入：
    // - 由 Unity Inspector 中的 Camera Sensor Component 提供
    //
    // 向量输入：
    // - 6 个关节位置
    // - 6 个关节速度
    //
    // 所以 Behavior Parameters 中：
    // Vector Observation Space Size = 12
    // Continuous Actions = 6
    //
    // 注意：
    // 这里不再添加焊缝相对坐标、路径进度、起点距离等辅助观测。
    // 焊缝几何信息只用于奖励计算，不直接作为观测输入。
    // ==========================================================

    public override void CollectObservations(VectorSensor sensor)
    {
        foreach (var joint in robotJoints)
        {
            if (joint == null)
            {
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
                continue;
            }

            float pos = 0f;
            float vel = 0f;

            if (joint.jointPosition.dofCount > 0)
            {
                pos = joint.jointPosition[0];
            }

            if (joint.jointVelocity.dofCount > 0)
            {
                vel = joint.jointVelocity[0];
            }

            if (float.IsNaN(pos) || float.IsInfinity(pos))
            {
                pos = 0f;
            }

            if (float.IsNaN(vel) || float.IsInfinity(vel))
            {
                vel = 0f;
            }

            var drive = joint.xDrive;

            float lower = drive.lowerLimit;
            float upper = drive.upperLimit;
            float range = upper - lower;

            float normalizedPos = 0f;

            if (Mathf.Abs(range) > 0.001f)
            {
                normalizedPos = Mathf.InverseLerp(lower, upper, pos) * 2f - 1f;
            }

            float normalizedVel = Mathf.Clamp(vel / 5f, -1f, 1f);

            sensor.AddObservation(normalizedPos);
            sensor.AddObservation(normalizedVel);
        }
    }

    // ===================== 动作执行与奖励计算 =====================

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        ApplyJointActions(actionBuffers);

        float downwardAlignment = GetDownwardAlignment();

        Vector3 nearestTarget = GetNearestSeamTarget(out float nearestT);
        float distError = Vector3.Distance(endEffector.position, nearestTarget);

        Vector3 startTarget = GetSeamTargetAtT(0f);
        Vector3 endTarget = GetSeamTargetAtT(1f);

        float distToStart = Vector3.Distance(endEffector.position, startTarget);
        float distToEnd = Vector3.Distance(endEffector.position, endTarget);

        // ===================== 失败终止条件 =====================

        if (downwardAlignment < MIN_DOWNWARD_ALIGNMENT)
        {
            phase = WeldingPhase.Failed;
            AddReward(-1.0f);
            EndEpisode();
            return;
        }

        if (distError > MAX_DISTANCE)
        {
            phase = WeldingPhase.Failed;
            AddReward(-1.0f);
            EndEpisode();
            return;
        }

        stagnationCounter++;

        if (stagnationCounter > STAGNATION_LIMIT)
        {
            phase = WeldingPhase.Failed;
            AddReward(-0.5f);
            EndEpisode();
            return;
        }

        // ===================== 基础奖励与惩罚 =====================

        AddReward(-0.0005f);

        AddJointMotionPenalty(actionBuffers);
        AddCameraVisibilityReward(nearestTarget);

        // ===================== 状态机 =====================

        switch (phase)
        {
            case WeldingPhase.ApproachStart:
                HandleApproachStart(distToStart, downwardAlignment);
                break;

            case WeldingPhase.TrackSeam:
                HandleTrackSeam(nearestT, distError, downwardAlignment, distToEnd);
                break;

            case WeldingPhase.Success:
            case WeldingPhase.Failed:
                break;
        }
    }

    private void ApplyJointActions(ActionBuffers actionBuffers)
    {
        var actions = actionBuffers.ContinuousActions;

        for (int i = 0; i < numJoints; i++)
        {
            if (robotJoints[i] == null) continue;

            var drive = robotJoints[i].xDrive;

            float action = Mathf.Clamp(actions[i], -1f, 1f);
            float newTarget = drive.target + action * actionScale;

            drive.target = Mathf.Clamp(newTarget, drive.lowerLimit, drive.upperLimit);
            robotJoints[i].xDrive = drive;
        }
    }

    private void HandleApproachStart(float distToStart, float downwardAlignment)
    {
        float approachReward = Mathf.Exp(-8f * distToStart) * 0.004f;
        AddReward(approachReward);

        float alignReward = Mathf.Clamp01(
            (downwardAlignment - MIN_DOWNWARD_ALIGNMENT) / (1f - MIN_DOWNWARD_ALIGNMENT)
        ) * 0.001f;

        AddReward(alignReward);

        if (distToStart <= startCaptureTolerance && downwardAlignment > 0.7f)
        {
            phase = WeldingPhase.TrackSeam;
            currentProgressT = 0f;
            stagnationCounter = 0;

            AddReward(0.2f);
        }
    }

    private void HandleTrackSeam(float nearestT, float distError, float downwardAlignment, float distToEnd)
    {
        float distanceReward = Mathf.Exp(-10f * distError) * 0.005f;
        AddReward(distanceReward);

        float alignReward = Mathf.Clamp01(
            (downwardAlignment - MIN_DOWNWARD_ALIGNMENT) / (1f - MIN_DOWNWARD_ALIGNMENT)
        ) * 0.002f;

        AddReward(alignReward);

        if (nearestT > currentProgressT + 0.001f)
        {
            float progressDelta = nearestT - currentProgressT;

            if (distError <= crossTrackTolerance && downwardAlignment > 0.7f)
            {
                float distanceQuality = 1f - Mathf.Clamp01(distError / crossTrackTolerance);
                float quality = distanceQuality * downwardAlignment;

                AddReward(progressDelta * 20f * quality);

                currentProgressT = nearestT;
                stagnationCounter = 0;
            }
            else if (distError < crossTrackTolerance * 2f)
            {
                currentProgressT = Mathf.Max(currentProgressT, nearestT);
            }
        }

        if (currentProgressT >= PROGRESS_COMPLETE)
        {
            if (distToEnd <= crossTrackTolerance)
            {
                phase = WeldingPhase.Success;
                AddReward(5.0f);
            }
            else
            {
                phase = WeldingPhase.Success;
                AddReward(1.0f);
            }

            EndEpisode();
        }
    }

    // ===================== 辅助奖励 =====================

    private void AddCameraVisibilityReward(Vector3 worldTarget)
    {
        if (weldCamera == null) return;

        Vector3 viewport = weldCamera.WorldToViewportPoint(worldTarget);

        bool inFront = viewport.z > 0f;
        bool inView =
            viewport.x >= 0f &&
            viewport.x <= 1f &&
            viewport.y >= 0f &&
            viewport.y <= 1f;

        if (!inFront)
        {
            AddReward(-0.001f);
            return;
        }

        if (inView)
        {
            AddReward(0.001f);

            Vector2 center = new Vector2(0.5f, 0.5f);
            Vector2 point = new Vector2(viewport.x, viewport.y);

            float centerError = Vector2.Distance(point, center);
            float centerReward = Mathf.Clamp01(1f - centerError * 2f) * cameraCenterRewardWeight;

            AddReward(centerReward);
        }
        else
        {
            AddReward(-0.0005f);
        }
    }

    private void AddJointMotionPenalty(ActionBuffers actionBuffers)
    {
        var actions = actionBuffers.ContinuousActions;

        float sumAbsAction = 0f;

        for (int i = 0; i < numJoints; i++)
        {
            sumAbsAction += Mathf.Abs(actions[i]);
        }

        float avgAbsAction = sumAbsAction / Mathf.Max(1, numJoints);

        AddReward(-avgAbsAction * jointMotionPenalty);
    }

    // ===================== Spline 计算 =====================

    private Vector3 GetNearestSeamTarget(out float t)
    {
        float3 localEndPos = seamSpline.transform.InverseTransformPoint(endEffector.position);

        SplineUtility.GetNearestPoint(
            seamSpline.Spline,
            localEndPos,
            out float3 localNearest,
            out t,
            4,
            2
        );

        Vector3 worldNearest = seamSpline.transform.TransformPoint(ToVector3(localNearest));
        Vector3 worldTarget = worldNearest + Vector3.up * standOffDistance;

        return worldTarget;
    }

    private Vector3 GetSeamTargetAtT(float t)
    {
        float3 localPoint = SplineUtility.EvaluatePosition(seamSpline.Spline, t);
        Vector3 worldPoint = seamSpline.transform.TransformPoint(ToVector3(localPoint));

        return worldPoint + Vector3.up * standOffDistance;
    }

    private Vector3 ToVector3(float3 value)
    {
        return new Vector3(value.x, value.y, value.z);
    }

    // ===================== 姿态判断 =====================

    private float GetDownwardAlignment()
    {
        return Vector3.Dot(endEffector.forward, Vector3.down);
    }

    // ===================== 手动测试控制 =====================

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var cont = actionsOut.ContinuousActions;

        for (int i = 0; i < cont.Length; i++)
        {
            cont[i] = 0f;
        }

        if (cont.Length > 0)
        {
            cont[0] = Input.GetAxis("Horizontal");
        }

        if (cont.Length > 1)
        {
            cont[1] = Input.GetAxis("Vertical");
        }
    }
}