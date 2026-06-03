using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class RobotArmAgent : Agent
{
    [Header("Original Grasping Task")]
    public Transform target;
    public Transform endEffector;
    public ArticulationBody[] joints;
    public Transform dropZone;

    [Header("Top-down Camera Vision")]
    public Camera topDownCamera;

    [Header("Vision Setting")]
    // 训练时建议关闭：直接使用 target.position，避免相机被机械臂/障碍物遮挡后视觉丢失
    public bool useVisionForTarget = false;

    // 是否显示紫色 Debug 小球
    public bool showDebugTargetSphere = true;

    [Header("Debug Target Sphere Display")]
    // Debug 小球和蓝色目标块之间的平面距离
    public float debugDisplayOffset = 0.16f;

    // Debug 小球底部贴在 Plane 上。
    // 如果你的 Debug 球 scale 是 0.04，那么半径约 0.02，所以这里默认 0.02。
    // 如果你想让球心严格等于 Plane 高度，可以改成 0。
    public float debugDisplayHeightOffset = 0.02f;

    // Debug 小球离蓝色目标块的最小距离
    public float debugDisplayTargetClearance = 0.10f;

    // Debug 小球离黄色 DropZone 的最小距离
    public float debugDisplayDropzoneClearance = 0.14f;

    // Debug 小球离 Obstacle 的最小距离
    public float debugDisplayObstacleClearance = 0.16f;

    private GameObject debugSphere;
    private Vector3 debugSphereFixedWorldPosition;

    private Vector3 predictedTargetPos;

    private Transform originalTargetParent;
    private Rigidbody targetRb;

    // 状态机：
    // 0 = 到目标块上方
    // 1 = 下降抓取
    // 2 = 搬运到放置区上方
    // 3 = 下降放置
    // 4 = 回到安全高度并进入下一轮
    private int currentState = 0;
    private int pauseCounter = 0;

    [Header("Obstacle Avoidance")]
    // 拖 Project 里的 ObstaclePrefab，不要拖 Hierarchy 里的场景物体
    public GameObject obstaclePrefab;

    // 拖 Hierarchy 里的 ObstacleParent
    public Transform obstacleParent;

    // 拖 SafetyPoint_Link3、SafetyPoint_Link5、SafetyPoint_Flange
    public Transform[] safetyPoints;

    // 重要：
    // maxObstacleCount = 1 时，Behavior Parameters 的 Vector Observation Space Size = 25
    // maxObstacleCount = 2 时，Behavior Parameters 的 Vector Observation Space Size = 32
    public int maxObstacleCount = 1;

    // 拖 Plane 的 Mesh Collider
    public Collider tableCollider;

    // 固定障碍物尺寸，防止训练时变成小方块
    public bool useFixedObstacleScale = true;

    // 推荐障碍物尺寸：比较像工业场景里的立柱/夹具
    public Vector3 fixedObstacleScale = new Vector3(0.16f, 0.35f, 0.16f);

    // 完成一轮抓取放置后，是否重新随机障碍物
    public bool randomizeObstacleAfterEachSuccessfulCycle = true;

    public bool enableDebugLogs = false;

    private readonly List<GameObject> obstacles = new List<GameObject>();

    [Header("Action Stability")]
    // 动作平滑系数：越小越稳，越大越灵敏
    public float actionSmoothFactor = 0.15f;

    // 是否把关节 target 限制在 URDF lower/upper limit 内
    public bool clampJointTargetsToLimits = true;

    private float[] smoothedActions = new float[6];

    [Header("Reward Memory")]
    private float lastDistanceToGoal = 0f;
    private float[] lastActions = new float[6];

    public override void Initialize()
    {
        originalTargetParent = target.parent;
        targetRb = target.GetComponent<Rigidbody>();
        predictedTargetPos = target.position;

        foreach (var joint in joints)
        {
            var drive = joint.xDrive;
            drive.stiffness = 100000f;
            drive.damping = 10000f;
            drive.forceLimit = 10000f;
            joint.xDrive = drive;
        }
    }

    private void ResetTargetVelocity()
    {
        if (targetRb == null)
        {
            return;
        }

        // Unity 不支持给 kinematic Rigidbody 设置 velocity
        if (targetRb.isKinematic)
        {
            return;
        }

#if UNITY_6000_0_OR_NEWER
        targetRb.linearVelocity = Vector3.zero;
#else
        targetRb.velocity = Vector3.zero;
#endif

        targetRb.angularVelocity = Vector3.zero;
    }

    private float GetTableTopY()
    {
        if (tableCollider != null)
        {
            return tableCollider.bounds.max.y;
        }

        if (originalTargetParent != null)
        {
            return originalTargetParent.position.y;
        }

        return 0f;
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

            if (randomPos.magnitude < 0.001f)
            {
                randomPos = Vector3.forward;
            }

            float minReach = 0.4f;
            float maxReach = 0.55f;
            randomDistance = Random.Range(minReach, maxReach);

            target.localPosition = randomPos.normalized * randomDistance + new Vector3(0f, targetHeight / 2f, 0f);

            Vector2 targetXZ = new Vector2(target.position.x, target.position.z);
            Vector2 dropZoneXZ = new Vector2(dropZone.position.x, dropZone.position.z);

            isInsideDropZoneColumn = Vector2.Distance(targetXZ, dropZoneXZ) < 0.25f;
        }
        while (isInsideDropZoneColumn);

        // 训练阶段建议不用视觉，直接使用真实 target.position
        if (useVisionForTarget && topDownCamera != null)
        {
            topDownCamera.Render();
            ExtractTargetCoordinatesFromImage();
        }
        else
        {
            predictedTargetPos = target.position;
        }

        // 注意：
        // 这里不直接更新 Debug 小球位置。
        // 因为 obstacle 会在 RespawnTarget() 后生成，
        // Debug 小球要等 PrepareObstacles() 之后再找安全显示位置。
    }

    private void ExtractTargetCoordinatesFromImage()
    {
        if (topDownCamera == null)
        {
            predictedTargetPos = target.position;
            return;
        }

        RenderTexture rt = topDownCamera.targetTexture;

        if (rt == null)
        {
            Debug.LogWarning("RenderTexture 未设置！");
            predictedTargetPos = target.position;
            return;
        }

        RenderTexture previousActive = RenderTexture.active;
        RenderTexture.active = rt;

        Texture2D tex = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
        tex.Apply();

        RenderTexture.active = previousActive;

        Color[] pixels = tex.GetPixels();

        double sumX = 0;
        double sumY = 0;
        int targetPixelCount = 0;

        // 只检测纯蓝色目标块
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

            // 视觉只估计 X/Z，Y 用目标块自身高度
            predictedTargetPos.y = target.position.y;
        }
        else
        {
            Debug.LogWarning("视觉丢失：没有找到蓝色像素。训练时建议关闭 Use Vision For Target。");
            predictedTargetPos = target.position;
        }
    }

    public override void OnEpisodeBegin()
    {
        currentState = 0;
        pauseCounter = 0;

        target.SetParent(originalTargetParent);

        Collider targetCollider = target.GetComponent<Collider>();
        if (targetCollider != null)
        {
            targetCollider.enabled = true;
        }

        if (targetRb != null)
        {
            targetRb.isKinematic = false;
            targetRb.constraints = RigidbodyConstraints.FreezePositionX |
                                   RigidbodyConstraints.FreezePositionZ |
                                   RigidbodyConstraints.FreezeRotation;

            ResetTargetVelocity();
        }

        // 先生成目标
        RespawnTarget();

        // 再生成障碍物，避免障碍物出现在目标附近
        PrepareObstacles();

        // 最后生成/刷新 Debug 显示点。
        // 这个点只在每次目标重新生成时更新一次，不会跟着蓝色块移动。
        PlaceDebugSphereForCurrentTarget();

        float[] homeAngles = new float[6] { 0f, -90f, 90f, 0f, 0f, 0f };

        for (int i = 0; i < joints.Length && i < homeAngles.Length; i++)
        {
            joints[i].SetDriveTarget(ArticulationDriveAxis.X, homeAngles[i]);
            joints[i].jointPosition = new ArticulationReducedSpace(homeAngles[i] * Mathf.Deg2Rad);
            joints[i].jointVelocity = new ArticulationReducedSpace(0f);
        }

        ResetProgressMemory();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // 训练时不使用视觉，始终使用真实目标位置
        if (!useVisionForTarget)
        {
            predictedTargetPos = target.position;
        }

        Vector3 predictedCurrentGoalPos = GetPredictedCurrentGoalPosition();
        Vector3 dirToPredictedGoal = predictedCurrentGoalPos - endEffector.position;

        // 原始抓取任务观测，大约 17 维
        sensor.AddObservation(dirToPredictedGoal.normalized);  // 3
        sensor.AddObservation(dirToPredictedGoal.magnitude);   // 1
        sensor.AddObservation(endEffector.forward);            // 3
        sensor.AddObservation(endEffector.up);                 // 3

        foreach (var joint in joints)
        {
            sensor.AddObservation(joint.jointPosition[0]);     // 通常 6
        }

        sensor.AddObservation((float)currentState);            // 1

        // Obstacle 观测：
        // 即使 obstacle_count = 0，也补零，保证维度不变
        for (int i = 0; i < maxObstacleCount; i++)
        {
            if (i < obstacles.Count && obstacles[i] != null && obstacles[i].activeSelf)
            {
                Vector3 relObstaclePos = obstacles[i].transform.position - endEffector.position;

                sensor.AddObservation(relObstaclePos / 1.0f);                       // 3
                sensor.AddObservation(obstacles[i].transform.localScale / 0.5f);    // 3
                sensor.AddObservation(1f);                                          // 1
            }
            else
            {
                sensor.AddObservation(Vector3.zero);   // 3
                sensor.AddObservation(Vector3.zero);   // 3
                sensor.AddObservation(0f);             // 1
            }
        }

        // 没有障碍物时返回 999，Clamp 后为 1，表示很安全
        float minObstacleDistance = Mathf.Clamp(GetMinimumObstacleDistance(), 0f, 1f);
        sensor.AddObservation(minObstacleDistance);     // 1

        // 这里不更新 Debug 小球。
        // Debug 小球只在 RespawnTarget + PrepareObstacles 后刷新一次。
    }

    private Vector3 GetCurrentGoalPosition()
    {
        float dropZoneTopY = dropZone.GetComponent<Collider>().bounds.max.y;

        BoxCollider targetCol = target.GetComponent<BoxCollider>();
        float targetHeight = targetCol.size.y * target.transform.lossyScale.y;

        float targetTopY = target.position.y + targetHeight / 2f;
        float flangeOffset = 0.01f;

        if (currentState == 0)
        {
            return new Vector3(target.position.x, targetTopY + 0.05f, target.position.z);
        }

        if (currentState == 1)
        {
            return new Vector3(target.position.x, targetTopY + flangeOffset, target.position.z);
        }

        if (currentState == 2)
        {
            return new Vector3(dropZone.position.x, dropZoneTopY + targetHeight + 0.15f, dropZone.position.z);
        }

        if (currentState == 3)
        {
            return new Vector3(dropZone.position.x, dropZoneTopY + targetHeight + flangeOffset, dropZone.position.z);
        }

        if (currentState == 4)
        {
            return new Vector3(dropZone.position.x, dropZoneTopY + targetHeight + 0.15f, dropZone.position.z);
        }

        return target.position;
    }

    private Vector3 GetPredictedCurrentGoalPosition()
    {
        float dropZoneTopY = dropZone.GetComponent<Collider>().bounds.max.y;

        BoxCollider targetCol = target.GetComponent<BoxCollider>();
        float targetHeight = targetCol.size.y * target.transform.lossyScale.y;

        float targetTopY = predictedTargetPos.y + targetHeight / 2f;
        float flangeOffset = 0.01f;

        if (currentState == 0)
        {
            return new Vector3(predictedTargetPos.x, targetTopY + 0.05f, predictedTargetPos.z);
        }

        if (currentState == 1)
        {
            return new Vector3(predictedTargetPos.x, targetTopY + flangeOffset, predictedTargetPos.z);
        }

        if (currentState == 2)
        {
            return new Vector3(dropZone.position.x, dropZoneTopY + targetHeight + 0.15f, dropZone.position.z);
        }

        if (currentState == 3)
        {
            return new Vector3(dropZone.position.x, dropZoneTopY + targetHeight + flangeOffset, dropZone.position.z);
        }

        if (currentState == 4)
        {
            return new Vector3(dropZone.position.x, dropZoneTopY + targetHeight + 0.15f, dropZone.position.z);
        }

        return predictedTargetPos;
    }

    private void EnsureDebugTargetSphere()
    {
        if (!showDebugTargetSphere)
        {
            return;
        }

        if (debugSphere != null)
        {
            return;
        }

        debugSphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        debugSphere.name = "Debug_PredictedTargetPosition";
        debugSphere.transform.localScale = new Vector3(0.04f, 0.04f, 0.04f);

        // 挂到当前环境下，避免多个 Debug 球堆在场景根目录
        if (originalTargetParent != null)
        {
            debugSphere.transform.SetParent(originalTargetParent);
        }

        Collider debugCol = debugSphere.GetComponent<Collider>();
        if (debugCol != null)
        {
            debugCol.enabled = false;
        }

        Renderer debugRenderer = debugSphere.GetComponent<Renderer>();
        if (debugRenderer != null)
        {
            debugRenderer.material.color = Color.magenta;
        }
    }

    private void PlaceDebugSphereForCurrentTarget()
    {
        if (!showDebugTargetSphere)
        {
            return;
        }

        EnsureDebugTargetSphere();

        if (debugSphere == null)
        {
            return;
        }

        debugSphereFixedWorldPosition = GetSafeDebugDisplayPositionOnPlane();
        debugSphere.transform.position = debugSphereFixedWorldPosition;
    }

    private Vector3 GetSafeDebugDisplayPositionOnPlane()
    {
        Vector3 center = predictedTargetPos;

        float displayY = GetTableTopY() + debugDisplayHeightOffset;
        float r = debugDisplayOffset;

        // 一组固定候选位置。
        // 它们都在蓝色目标块周围，但不会放在蓝色目标块正上方。
        Vector3[] candidateOffsets = new Vector3[]
        {
            new Vector3( r, 0f,  0f),
            new Vector3(-r, 0f,  0f),
            new Vector3( 0f, 0f,  r),
            new Vector3( 0f, 0f, -r),

            new Vector3( 0.7f * r, 0f,  0.7f * r),
            new Vector3(-0.7f * r, 0f,  0.7f * r),
            new Vector3( 0.7f * r, 0f, -0.7f * r),
            new Vector3(-0.7f * r, 0f, -0.7f * r),

            new Vector3( 1.4f * r, 0f,  0f),
            new Vector3(-1.4f * r, 0f,  0f),
            new Vector3( 0f, 0f,  1.4f * r),
            new Vector3( 0f, 0f, -1.4f * r),
        };

        foreach (Vector3 offset in candidateOffsets)
        {
            Vector3 candidate = new Vector3(
                center.x + offset.x,
                displayY,
                center.z + offset.z
            );

            if (IsValidDebugDisplayPosition(candidate))
            {
                return candidate;
            }
        }

        // 如果都不合适，就放在目标右前方更远的位置
        return new Vector3(
            center.x + 0.25f,
            displayY,
            center.z + 0.25f
        );
    }

    private bool IsValidDebugDisplayPosition(Vector3 candidate)
    {
        Vector2 candidateXZ = new Vector2(candidate.x, candidate.z);

        // 1. 不要和蓝色目标块太近
        Vector2 targetXZ = new Vector2(predictedTargetPos.x, predictedTargetPos.z);
        if (Vector2.Distance(candidateXZ, targetXZ) < debugDisplayTargetClearance)
        {
            return false;
        }

        // 2. 不要和黄色 DropZone 太近
        if (dropZone != null)
        {
            Vector2 dropXZ = new Vector2(dropZone.position.x, dropZone.position.z);

            if (Vector2.Distance(candidateXZ, dropXZ) < debugDisplayDropzoneClearance)
            {
                return false;
            }
        }

        // 3. 不要和 Obstacle 太近
        foreach (GameObject obs in obstacles)
        {
            if (obs == null || !obs.activeSelf)
            {
                continue;
            }

            Vector2 obsXZ = new Vector2(obs.transform.position.x, obs.transform.position.z);

            if (Vector2.Distance(candidateXZ, obsXZ) < debugDisplayObstacleClearance)
            {
                return false;
            }
        }

        return true;
    }

    private void PrepareObstacles()
    {
        int activeObstacleCount = Mathf.RoundToInt(
            Academy.Instance.EnvironmentParameters.GetWithDefault("obstacle_count", 1f)
        );

        activeObstacleCount = Mathf.Clamp(activeObstacleCount, 0, maxObstacleCount);

        if (obstaclePrefab == null)
        {
            if (activeObstacleCount > 0)
            {
                Debug.LogWarning("Obstacle Prefab 没有设置，但 obstacle_count > 0。");
            }

            return;
        }

        while (obstacles.Count < maxObstacleCount)
        {
            Transform parent = obstacleParent != null ? obstacleParent : originalTargetParent;

            GameObject obs = Instantiate(obstaclePrefab, parent);
            obs.name = "Generated_Obstacle_" + obstacles.Count;

            try
            {
                obs.tag = "Obstacle";
            }
            catch
            {
                Debug.LogWarning("请先在 Unity 的 Tags 中创建 Obstacle 标签。");
            }

            Rigidbody rb = obs.GetComponent<Rigidbody>();
            if (rb == null)
            {
                rb = obs.AddComponent<Rigidbody>();
            }

            rb.isKinematic = true;
            rb.useGravity = false;

            obstacles.Add(obs);
        }

        for (int i = 0; i < obstacles.Count; i++)
        {
            bool active = i < activeObstacleCount;
            obstacles[i].SetActive(active);

            if (active)
            {
                RandomizeObstacle(obstacles[i]);
            }
        }
    }

    private void RandomizeObstacle(GameObject obs)
    {
        float tableTopY = GetTableTopY();

        float sx;
        float sy;
        float sz;

        if (useFixedObstacleScale)
        {
            sx = Academy.Instance.EnvironmentParameters.GetWithDefault("obstacle_scale_x", fixedObstacleScale.x);
            sy = Academy.Instance.EnvironmentParameters.GetWithDefault("obstacle_scale_y", fixedObstacleScale.y);
            sz = Academy.Instance.EnvironmentParameters.GetWithDefault("obstacle_scale_z", fixedObstacleScale.z);
        }
        else
        {
            sx = Random.Range(0.12f, 0.20f);
            sy = Random.Range(0.25f, 0.40f);
            sz = Random.Range(0.12f, 0.20f);
        }

        obs.transform.localScale = new Vector3(sx, sy, sz);

        float minX = Academy.Instance.EnvironmentParameters.GetWithDefault("obstacle_min_x", -0.45f);
        float maxX = Academy.Instance.EnvironmentParameters.GetWithDefault("obstacle_max_x", 0.45f);
        float minZ = Academy.Instance.EnvironmentParameters.GetWithDefault("obstacle_min_z", 0.20f);
        float maxZ = Academy.Instance.EnvironmentParameters.GetWithDefault("obstacle_max_z", 0.75f);

        Vector3 chosenWorldPos = Vector3.zero;
        bool foundValidPosition = false;

        for (int attempt = 0; attempt < 200; attempt++)
        {
            float x = Random.Range(minX, maxX);
            float z = Random.Range(minZ, maxZ);

            Vector3 candidateLocalPos = new Vector3(x, 0f, z);
            Vector3 candidateWorldPos = originalTargetParent.TransformPoint(candidateLocalPos);

            if (IsValidObstaclePosition(candidateWorldPos, obs))
            {
                chosenWorldPos = candidateWorldPos;
                foundValidPosition = true;
                break;
            }
        }

        if (!foundValidPosition)
        {
            Vector3[] fallbackLocalPositions = new Vector3[]
            {
                new Vector3(0.35f, 0f, 0.55f),
                new Vector3(-0.35f, 0f, 0.55f),
                new Vector3(0.40f, 0f, 0.35f),
                new Vector3(-0.40f, 0f, 0.35f)
            };

            foreach (Vector3 localPos in fallbackLocalPositions)
            {
                Vector3 candidateWorldPos = originalTargetParent.TransformPoint(localPos);

                if (IsValidObstaclePosition(candidateWorldPos, obs))
                {
                    chosenWorldPos = candidateWorldPos;
                    foundValidPosition = true;
                    break;
                }
            }
        }

        if (!foundValidPosition)
        {
            Vector3 fallbackLocalPos = new Vector3(0.35f, 0f, 0.55f);
            chosenWorldPos = originalTargetParent.TransformPoint(fallbackLocalPos);

            if (enableDebugLogs)
            {
                Debug.LogWarning("没有找到合法 obstacle 位置，使用 fallback 位置。");
            }
        }

        // Cube 的 Transform Position 是中心点
        chosenWorldPos.y = tableTopY + sy / 2f;

        obs.transform.position = chosenWorldPos;
        obs.transform.rotation = Quaternion.Euler(0f, Random.Range(0f, 180f), 0f);
    }

    private bool IsValidObstaclePosition(Vector3 worldPos, GameObject self)
    {
        float minDistToTarget = Academy.Instance.EnvironmentParameters.GetWithDefault("obstacle_target_clearance", 0.35f);
        float minDistToDropZone = Academy.Instance.EnvironmentParameters.GetWithDefault("obstacle_drop_clearance", 0.24f);
        float minDistToBase = Academy.Instance.EnvironmentParameters.GetWithDefault("obstacle_base_clearance", 0.28f);
        float minObstacleSpacing = Academy.Instance.EnvironmentParameters.GetWithDefault("obstacle_min_spacing", 0.24f);

        Vector2 candidateXZ = new Vector2(worldPos.x, worldPos.z);

        Vector2 targetXZ = new Vector2(target.position.x, target.position.z);
        Vector2 dropXZ = new Vector2(dropZone.position.x, dropZone.position.z);
        Vector2 baseXZ = new Vector2(originalTargetParent.position.x, originalTargetParent.position.z);

        if (Vector2.Distance(candidateXZ, targetXZ) < minDistToTarget)
        {
            return false;
        }

        if (Vector2.Distance(candidateXZ, dropXZ) < minDistToDropZone)
        {
            return false;
        }

        if (Vector2.Distance(candidateXZ, baseXZ) < minDistToBase)
        {
            return false;
        }

        foreach (GameObject obs in obstacles)
        {
            if (obs == null || obs == self || !obs.activeSelf)
            {
                continue;
            }

            Vector2 otherXZ = new Vector2(obs.transform.position.x, obs.transform.position.z);

            if (Vector2.Distance(candidateXZ, otherXZ) < minObstacleSpacing)
            {
                return false;
            }
        }

        return true;
    }

    private float GetMinimumObstacleDistance()
    {
        float minDistance = 999f;

        if (safetyPoints == null || safetyPoints.Length == 0)
        {
            return minDistance;
        }

        foreach (Transform p in safetyPoints)
        {
            if (p == null)
            {
                continue;
            }

            foreach (GameObject obs in obstacles)
            {
                if (obs == null || !obs.activeSelf)
                {
                    continue;
                }

                Collider col = obs.GetComponent<Collider>();
                if (col == null)
                {
                    continue;
                }

                Vector3 closestPoint = col.ClosestPoint(p.position);
                float d = Vector3.Distance(p.position, closestPoint);

                if (d < minDistance)
                {
                    minDistance = d;
                }
            }
        }

        return minDistance;
    }

    private bool ApplyObstacleAvoidanceReward()
    {
        float minDistance = GetMinimumObstacleDistance();

        // obstacle_count = 0 或 safetyPoints 没设置时，跳过避障惩罚
        if (minDistance > 900f)
        {
            return false;
        }

        float safeDistance = Academy.Instance.EnvironmentParameters.GetWithDefault("safe_distance", 0.20f);
        float collisionDistance = Academy.Instance.EnvironmentParameters.GetWithDefault("collision_distance", 0.045f);
        float obstaclePenaltyScale = Academy.Instance.EnvironmentParameters.GetWithDefault("obstacle_penalty_scale", 0.02f);

        if (minDistance < collisionDistance)
        {
            SetReward(-70.0f);
            EndEpisode();
            return true;
        }

        if (minDistance < safeDistance)
        {
            float danger = (safeDistance - minDistance) / safeDistance;
            AddReward(-obstaclePenaltyScale * danger * danger);
        }

        return false;
    }

    private void ResetProgressMemory()
    {
        lastDistanceToGoal = Vector3.Distance(endEffector.position, GetCurrentGoalPosition());

        for (int i = 0; i < lastActions.Length; i++)
        {
            lastActions[i] = 0f;
        }

        for (int i = 0; i < smoothedActions.Length; i++)
        {
            smoothedActions[i] = 0f;
        }
    }

    private void ChangeState(int newState)
    {
        currentState = newState;
        pauseCounter = 0;

        // 状态变化后目标点跳变，所以重置 progress 记忆
        lastDistanceToGoal = Vector3.Distance(endEffector.position, GetCurrentGoalPosition());

        // 这里不更新 Debug 小球。
        // Debug 小球只在蓝色目标块重新生成时更新一次。
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        if (pauseCounter == 0)
        {
            float actionMagnitude = 0f;
            float actionSmoothPenalty = 0f;

            int actionCount = Mathf.Min(6, joints.Length, actionBuffers.ContinuousActions.Length);

            for (int i = 0; i < actionCount; i++)
            {
                float rawAction = Mathf.Clamp(actionBuffers.ContinuousActions[i], -1f, 1f);

                actionMagnitude += Mathf.Abs(rawAction);

                if (i < lastActions.Length)
                {
                    actionSmoothPenalty += Mathf.Abs(rawAction - lastActions[i]);
                    lastActions[i] = rawAction;
                }

                // 动作平滑，减少训练初期抽搐
                smoothedActions[i] = Mathf.Lerp(
                    smoothedActions[i],
                    rawAction,
                    actionSmoothFactor
                );

                float actionStepDeg = Academy.Instance.EnvironmentParameters.GetWithDefault("action_step_deg", 0.35f);
                float deltaAngle = smoothedActions[i] * actionStepDeg;

                ArticulationDrive drive = joints[i].xDrive;
                float currentTarget = drive.target;
                float newTarget = currentTarget + deltaAngle;

                if (clampJointTargetsToLimits && drive.lowerLimit < drive.upperLimit)
                {
                    float margin = 2.0f;

                    newTarget = Mathf.Clamp(
                        newTarget,
                        drive.lowerLimit + margin,
                        drive.upperLimit - margin
                    );
                }

                joints[i].SetDriveTarget(ArticulationDriveAxis.X, newTarget);
            }

            AddReward(-actionMagnitude * 0.0001f);
            AddReward(-actionSmoothPenalty * 0.0002f);
        }

        float currentReachThreshold = Academy.Instance.EnvironmentParameters.GetWithDefault("reach_threshold", 0.05f);

        Vector3 offsetGoalPos = GetCurrentGoalPosition();
        float distanceToGoal = Vector3.Distance(endEffector.position, offsetGoalPos);

        float alignmentPenalty = 1.0f - Vector3.Dot(endEffector.forward, Vector3.down);

        float progressRewardScale = Academy.Instance.EnvironmentParameters.GetWithDefault("progress_reward_scale", 0.03f);
        float progress = lastDistanceToGoal - distanceToGoal;

        AddReward(progress * progressRewardScale);

        lastDistanceToGoal = distanceToGoal;

        float stepPenalty = -distanceToGoal * 0.01f - alignmentPenalty * 0.005f;

        // 目标块掉落或飞出工作空间
        Vector3 envLocalTargetPos = originalTargetParent.InverseTransformPoint(target.position);
        Vector2 targetXZ = new Vector2(envLocalTargetPos.x, envLocalTargetPos.z);

        if (envLocalTargetPos.y < -0.1f || targetXZ.magnitude > 0.9f)
        {
            SetReward(-100.0f);
            EndEpisode();
            return;
        }

        // 避障奖励
        if (ApplyObstacleAvoidanceReward())
        {
            return;
        }

        Vector2 effXZ = new Vector2(endEffector.position.x, endEffector.position.z);

        switch (currentState)
        {
            case 0:
                if (distanceToGoal < currentReachThreshold * 1.5f && alignmentPenalty < 0.1f)
                {
                    pauseCounter++;

                    if (pauseCounter > 25)
                    {
                        ChangeState(1);
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

                if (distanceToGoal < currentReachThreshold &&
                    alignmentPenalty < 0.1f &&
                    xzDeviation < currentReachThreshold)
                {
                    pauseCounter++;
                    AddReward(0.01f);

                    float requiredGrabFrames = Academy.Instance.EnvironmentParameters.GetWithDefault("grab_pause_frames", 50.0f);

                    if (pauseCounter >= requiredGrabFrames)
                    {
                        if (targetRb != null)
                        {
                            targetRb.isKinematic = true;
                        }

                        Collider targetCollider = target.GetComponent<Collider>();
                        if (targetCollider != null)
                        {
                            targetCollider.enabled = false;
                        }

                        target.SetParent(endEffector);

                        BoxCollider targetCol = target.GetComponent<BoxCollider>();
                        float targetHeight = targetCol.size.y * target.transform.lossyScale.y;

                        target.position = endEffector.position + endEffector.forward * (targetHeight / 2f + 0.01f);

                        AddReward(30.0f);

                        ChangeState(2);
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

                    if (pauseCounter > 25)
                    {
                        ChangeState(3);
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

                if (distanceToGoal < currentReachThreshold &&
                    alignmentPenalty < 0.1f &&
                    dropXzDeviation < currentReachThreshold)
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
                            ResetTargetVelocity();
                        }

                        Collider targetCollider = target.GetComponent<Collider>();
                        if (targetCollider != null)
                        {
                            targetCollider.enabled = true;
                        }

                        ChangeState(4);
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
                        ResetTargetVelocity();
                    }

                    RespawnTarget();

                    if (randomizeObstacleAfterEachSuccessfulCycle)
                    {
                        PrepareObstacles();
                    }

                    // 每次目标块重新生成后，刷新一次 Debug 小球位置。
                    // 它不会跟着目标块移动，只会在下一次 Respawn 时更新。
                    PlaceDebugSphereForCurrentTarget();

                    if (targetRb != null)
                    {
                        targetRb.isKinematic = false;
                        targetRb.constraints = RigidbodyConstraints.FreezePositionX |
                                               RigidbodyConstraints.FreezePositionZ |
                                               RigidbodyConstraints.FreezeRotation;
                    }

                    ChangeState(0);
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

        for (int i = 2; i < 6; i++)
        {
            continuousActionsOut[i] = 0f;
        }
    }
}