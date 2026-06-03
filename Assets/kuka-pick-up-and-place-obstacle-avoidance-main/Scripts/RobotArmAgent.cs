using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

using UnityEngine;
using UnityEngine.Serialization;

public class RobotArmAgent : Agent
{
    [Header("References")]
    [Tooltip("Pose reference for observations and alignment (e.g. flange). Must match the trained model's coordinate frame.")]
    public Transform endEffector;
    [Tooltip("Attachment point for gripping (e.g. tool0). Falls back to endEffector if unset.")]
    public Transform gripAnchor;
    public ArticulationBody[] joints;
    [FormerlySerializedAs("dropOffArea")]
    public Transform dropZone;
    public RandomPickupPartSpawner partSpawner;
    [Tooltip("Workspace camera rig that captures the global pick-up/drop-off view once per episode.")]
    public WorkspaceCameraRig workspaceCameraRig;

    [Header("Flange Orientation")]
    public Vector3 flangeApproachAxis = Vector3.forward;

    [Header("Grip Settings")]
    public float attachDistance = 0.05f;
    public float alignmentThreshold = 0.85f;
    public float heldPartClearance = 0.0025f;
    public int defaultGrabPauseFrames = 60;
    public int defaultDropPauseFrames = 60;

    [Header("Goal Heights")]
    public float pickupHoverHeight = 0.24f;
    public float pickupContactClearance = 0.04f;
    public float dropHoverHeight = 0.15f;
    public float dropReleaseClearance = 0.012f;
    public float hoverHorizontalTolerance = 0.045f;
    public float hoverVerticalTolerance = 0.04f;
    public float carryActionScaleMultiplier = 0.65f;

    [Header("Drive Settings")]
    public float driveStiffness = 100000f;
    public float driveDamping = 10000f;
    public float driveForceLimit = 10000f;

    [Header("Rewards")]
    public float pickReward = 30f;
    public float placeReward = 100f;
    public float outOfBoundsPenalty = -20f;
    public float driftFailPenalty = -1f;
    public float actionSmoothScale = 0.001f;
    public float reachProgressScale = 2.0f;
    public float carryProgressScale = 2.0f;
    public float alignmentPenaltyScale = 0.0025f;
    public float idleStepPenalty = 0.0005f;
    public float hoverFrameReward = 0.01f;

    [Header("Obstacle")]
    public string obstacleTag = "Obstacle";
    public float obstacleHitPenalty = -10f;
    public bool endEpisodeOnObstacleHit = true;
    [Tooltip("Distance (m) at which a soft clearance penalty starts ramping. 0 disables.")]
    public float clearanceDistance = 0.08f;
    [Tooltip("Peak per-step penalty when end-effector is touching an obstacle (quadratic ramp).")]
    public float clearancePenaltyScale = 0.08f;

    [Header("Phase Bonuses")]
    [Tooltip("One-time reward per episode when first entering Drop Hover (state 4).")]
    public float carryToDropBonus = 10f;
    [Tooltip("One-time reward per episode when first entering Descend & Release (state 5).")]
    public float dropToReleaseBonus = 15f;

    [Header("Episode Control")]
    public int defaultMaxStep = 0;
    public float progressEpsilon = 0.001f;
    public int stagnationStepLimit = 300;
    public int hoverStateTimeoutSteps = 400;

    private Transform target;
    private Rigidbody targetRb;
    private Collider targetCollider;
    private Transform originalTargetParent;
    private int currentState;
    private int maxStateReached;
    private int pauseCounter;
    private int hoverStateSteps;
    private readonly float[] homeAngles = { 0f, -90f, 90f, 0f, 0f, 0f };
    private Vector3 rootInitialPos;
    private Quaternion rootInitialRot;
    private ArticulationBody articulationRoot;
    private float previousGoalDistance;
    private int stagnationSteps;
    private Collider[] robotColliders;
    private Collider[] cachedObstacles;
    private bool holdingCollisionIgnored;
    private float targetHalfHeight = 0.02f;
    private bool referencesResolved;
    private StatsRecorder statsRecorder;
    private float currentReachProgressScale;
    private float currentCarryProgressScale;

    public override void Initialize()
    {
        EnsureReferencesResolved(true);

        if (MaxStep <= 0)
        {
            MaxStep = Mathf.RoundToInt(
                Academy.Instance.EnvironmentParameters.GetWithDefault("max_step", defaultMaxStep));
        }

        articulationRoot = joints[0].transform.root.GetComponent<ArticulationBody>();
        if (articulationRoot == null)
        {
            articulationRoot = joints[0].transform.root.GetComponentInChildren<ArticulationBody>();
        }

        rootInitialPos = articulationRoot.transform.position;
        rootInitialRot = articulationRoot.transform.rotation;
        robotColliders = articulationRoot.GetComponentsInChildren<Collider>(true);
        AttachObstacleForwarders(robotColliders);
        CacheObstacles();

        foreach (ArticulationBody joint in joints)
        {
            ArticulationDrive drive = joint.xDrive;
            drive.stiffness = driveStiffness;
            drive.damping = driveDamping;
            drive.forceLimit = driveForceLimit;
            joint.xDrive = drive;
        }

        statsRecorder = Academy.Instance.StatsRecorder;
    }

    public override void OnEpisodeBegin()
    {
        EnsureReferencesResolved(true);

        currentState = 0;
        maxStateReached = 0;
        pauseCounter = 0;
        stagnationSteps = 0;
        hoverStateSteps = 0;

        currentReachProgressScale = Academy.Instance.EnvironmentParameters.GetWithDefault(
            "approach_scale", reachProgressScale);
        currentCarryProgressScale = Academy.Instance.EnvironmentParameters.GetWithDefault(
            "carry_scale", carryProgressScale);

        if (target != null)
        {
            SetHeldCollisionIgnore(false);
            ReparentPreservingScale(target, originalTargetParent);

            if (targetRb != null)
            {
                targetRb.isKinematic = false;
                targetRb.linearVelocity = Vector3.zero;
                targetRb.angularVelocity = Vector3.zero;
            }
        }

        target = null;
        targetRb = null;
        targetCollider = null;
        targetHalfHeight = 0.02f;
        holdingCollisionIgnored = false;

        articulationRoot.TeleportRoot(rootInitialPos, rootInitialRot);
        List<float> positions = new List<float>();
        int dof = articulationRoot.GetJointPositions(positions);
        for (int i = 0; i < dof; i++)
        {
            positions[i] = 0f;
        }

        articulationRoot.SetJointPositions(positions);
        articulationRoot.SetJointVelocities(new List<float>(new float[dof]));

        for (int i = 0; i < joints.Length; i++)
        {
            ArticulationDrive drive = joints[i].xDrive;
            drive.stiffness = driveStiffness;
            drive.damping = driveDamping;
            drive.forceLimit = driveForceLimit;
            drive.target = homeAngles[i];
            joints[i].xDrive = drive;
        }

        RespawnAndAcquire();
        previousGoalDistance = GetDistanceToCurrentGoal();

        // One snapshot per episode — the agent must infer pick-up/drop-off layout from the image
        // for the remainder of the episode (no direct distance/position observations).
        if (workspaceCameraRig != null)
        {
            workspaceCameraRig.CaptureSnapshot();
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        EnsureReferencesResolved();

        // Proprioception only: joint angles, joint velocities, and a state one-hot.
        // The spatial layout (target pose, drop pose, end-effector pose) is provided through
        // the WorkspaceCameraRig snapshot consumed by RenderTextureSensorComponent.
        foreach (ArticulationBody joint in joints)
        {
            sensor.AddObservation(joint.jointPosition[0]);
        }

        foreach (ArticulationBody joint in joints)
        {
            sensor.AddObservation(joint.jointVelocity[0] / 6f);
        }

        for (int s = 0; s < 6; s++)
        {
            sensor.AddObservation(currentState == s ? 1f : 0f);
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        EnsureReferencesResolved();

        if (target == null)
        {
            statsRecorder.Add("Custom/EndReason/NoTarget", 1f, StatAggregationMethod.Sum);
            EndEpisode();
            return;
        }

        float drift = Vector3.Distance(articulationRoot.transform.position, rootInitialPos);
        if (drift > 0.5f)
        {
            statsRecorder.Add("Custom/EndReason/RootDrift", 1f, StatAggregationMethod.Sum);
            SetReward(driftFailPenalty);
            EndEpisode();
            return;
        }

        int jointCount = Mathf.Min(joints.Length, actionBuffers.ContinuousActions.Length);
        float actionStepScale = currentState >= 3 ? carryActionScaleMultiplier : 1f;
        float actionMagSq = 0f;
        for (int i = 0; i < jointCount; i++)
        {
            float actionVal = Mathf.Clamp(actionBuffers.ContinuousActions[i], -1f, 1f);
            actionMagSq += actionVal * actionVal;

            float deltaAngle = actionVal * 2f * actionStepScale;
            ArticulationDrive drive = joints[i].xDrive;
            float newTarget = Mathf.Clamp(drive.target + deltaAngle, drive.lowerLimit, drive.upperLimit);
            joints[i].SetDriveTarget(ArticulationDriveAxis.X, newTarget);
        }

        Vector3 offsetGoal = GetCurrentGoalPosition();
        float distanceToGoal = Vector3.Distance(endEffector.position, offsetGoal);
        float horizontalDistanceToGoal = GetHorizontalDistance(endEffector.position, offsetGoal);
        float verticalDistanceToGoal = Mathf.Abs(endEffector.position.y - offsetGoal.y);

        Vector3 worldApproach = endEffector.TransformDirection(flangeApproachAxis);
        float alignment = Vector3.Dot(worldApproach, Vector3.down);
        float misalignment = Mathf.Clamp01(1f - alignment);
        float progressDelta = previousGoalDistance - distanceToGoal;
        float shapingPenalty = -idleStepPenalty
            - (misalignment * alignmentPenaltyScale)
            - (actionMagSq * actionSmoothScale);

        ApplyClearanceShaping();

        float requiredGrabFrames = Academy.Instance.EnvironmentParameters.GetWithDefault(
            "grab_pause_frames", defaultGrabPauseFrames);
        float requiredDropFrames = Academy.Instance.EnvironmentParameters.GetWithDefault(
            "drop_pause_frames", defaultDropPauseFrames);

        // State 0: Reach — move to pickup hover position above target
        if (currentState == 0)
        {
            hoverStateSteps = 0;

            if (targetRb != null && !targetRb.isKinematic && targetRb.linearVelocity.magnitude > 0.1f)
            {
                AddReward(-targetRb.linearVelocity.magnitude * 0.05f);
            }

            if (horizontalDistanceToGoal < hoverHorizontalTolerance
                && verticalDistanceToGoal < (hoverVerticalTolerance * 1.5f)
                && alignment >= alignmentThreshold)
            {
                currentState = 1;
                pauseCounter = 0;
                stagnationSteps = 0;
                hoverStateSteps = 0;
                previousGoalDistance = GetDistanceToCurrentGoal();
            }
            else
            {
                AddReward(shapingPenalty);
                AddReward(Mathf.Clamp(progressDelta * currentReachProgressScale, -0.05f, 0.05f));
                if (TrackStagnation(progressDelta))
                {
                    return;
                }
            }
        }
        // State 1: Pickup Hover — hold above target for grab_pause_frames, then descend
        else if (currentState == 1)
        {
            hoverStateSteps++;

            if (horizontalDistanceToGoal < (hoverHorizontalTolerance * 0.85f)
                && verticalDistanceToGoal < (hoverVerticalTolerance * 1.15f)
                && alignment >= (alignmentThreshold - 0.1f))
            {
                pauseCounter++;
                AddReward(hoverFrameReward);

                if (pauseCounter >= requiredGrabFrames)
                {
                    currentState = 2;
                    pauseCounter = 0;
                    stagnationSteps = 0;
                    hoverStateSteps = 0;
                    previousGoalDistance = GetDistanceToCurrentGoal();
                }
            }
            else
            {
                pauseCounter = Mathf.Max(pauseCounter - 2, 0);
                AddReward(shapingPenalty);
                AddReward(Mathf.Clamp(progressDelta * currentReachProgressScale, -0.05f, 0.05f));

                bool shouldFallback = horizontalDistanceToGoal > (hoverHorizontalTolerance * 3.5f)
                    || endEffector.position.y > GetPickupHoverPosition().y + (hoverVerticalTolerance * 3f)
                    || hoverStateSteps >= hoverStateTimeoutSteps;

                if (shouldFallback)
                {
                    currentState = 0;
                    stagnationSteps = 0;
                    hoverStateSteps = 0;
                    previousGoalDistance = GetDistanceToCurrentGoal();
                    AddReward(-0.15f);
                }
                else if (TrackStagnation(progressDelta))
                {
                    return;
                }
            }
        }
        // State 2: Descend & Grab — move down to contact, attach on arrival
        else if (currentState == 2)
        {
            hoverStateSteps = 0;

            if (horizontalDistanceToGoal < (hoverHorizontalTolerance * 1.5f)
                && verticalDistanceToGoal < (hoverVerticalTolerance * 2.0f)
                && alignment >= (alignmentThreshold - 0.15f))
            {
                if (targetRb != null)
                {
                    targetRb.linearVelocity = Vector3.zero;
                    targetRb.angularVelocity = Vector3.zero;
                    targetRb.isKinematic = true;
                }

                Vector3 savedWorldScale = target.lossyScale;
                Transform anchor = gripAnchor != null ? gripAnchor : endEffector;
                target.SetParent(anchor, false);
                target.localPosition = GetHeldLocalPosition();
                target.localRotation = Quaternion.identity;
                Vector3 anchorScale = anchor.lossyScale;
                target.localScale = new Vector3(
                    savedWorldScale.x / anchorScale.x,
                    savedWorldScale.y / anchorScale.y,
                    savedWorldScale.z / anchorScale.z);
                SetHeldCollisionIgnore(true);

                currentState = 3;
                pauseCounter = 0;
                stagnationSteps = 0;
                previousGoalDistance = GetDistanceToCurrentGoal();
                AddReward(pickReward);
                statsRecorder.Add("Custom/PickSuccess", 1f, StatAggregationMethod.Sum);
            }
            else
            {
                AddReward(shapingPenalty);
                AddReward(Mathf.Clamp(progressDelta * currentReachProgressScale, -0.05f, 0.05f));
                if (TrackStagnation(progressDelta))
                {
                    return;
                }
            }
        }
        // State 3: Carry — move attached part toward drop hover position
        else if (currentState == 3)
        {
            hoverStateSteps = 0;

            if (horizontalDistanceToGoal < (hoverHorizontalTolerance * 1.2f)
                && verticalDistanceToGoal < (hoverVerticalTolerance * 1.8f)
                && alignment >= alignmentThreshold)
            {
                currentState = 4;
                pauseCounter = 0;
                stagnationSteps = 0;
                hoverStateSteps = 0;
                previousGoalDistance = GetDistanceToCurrentGoal();
                AwardStateAdvance(4);
            }
            else
            {
                AddReward(shapingPenalty);
                AddReward(Mathf.Clamp(progressDelta * currentCarryProgressScale, -0.05f, 0.05f));
                if (TrackStagnation(progressDelta))
                {
                    return;
                }
            }
        }
        // State 4: Drop Hover — hold above drop zone for drop_pause_frames, then descend
        else if (currentState == 4)
        {
            hoverStateSteps++;

            if (horizontalDistanceToGoal < (hoverHorizontalTolerance * 0.75f)
                && verticalDistanceToGoal < (hoverVerticalTolerance * 1.2f)
                && alignment >= (alignmentThreshold - 0.05f))
            {
                pauseCounter++;
                AddReward(hoverFrameReward);

                if (pauseCounter >= requiredDropFrames)
                {
                    currentState = 5;
                    pauseCounter = 0;
                    stagnationSteps = 0;
                    hoverStateSteps = 0;
                    previousGoalDistance = GetDistanceToCurrentGoal();
                    AwardStateAdvance(5);
                }
            }
            else
            {
                pauseCounter = Mathf.Max(pauseCounter - 2, 0);
                AddReward(shapingPenalty);
                AddReward(Mathf.Clamp(progressDelta * currentCarryProgressScale, -0.05f, 0.05f));

                bool shouldFallback = horizontalDistanceToGoal > (hoverHorizontalTolerance * 3.5f)
                    || endEffector.position.y > GetDropHoverPosition().y + (hoverVerticalTolerance * 3f)
                    || hoverStateSteps >= hoverStateTimeoutSteps;

                if (shouldFallback)
                {
                    currentState = 3;
                    stagnationSteps = 0;
                    hoverStateSteps = 0;
                    previousGoalDistance = GetDistanceToCurrentGoal();
                    AddReward(-0.15f);
                }
                else if (TrackStagnation(progressDelta))
                {
                    return;
                }
            }
        }
        // State 5: Descend & Release — move down to release position, drop and end episode
        else if (currentState == 5)
        {
            hoverStateSteps = 0;

            if (horizontalDistanceToGoal < (hoverHorizontalTolerance * 0.85f)
                && verticalDistanceToGoal < (hoverVerticalTolerance * 1.2f)
                && alignment >= (alignmentThreshold - 0.05f))
            {
                AddReward(placeReward);
                statsRecorder.Add("Custom/PlaceSuccess", 1f, StatAggregationMethod.Sum);
                statsRecorder.Add("Custom/EndReason/PlaceSuccess", 1f, StatAggregationMethod.Sum);

                SetHeldCollisionIgnore(false);
                ReparentPreservingScale(target, originalTargetParent);

                if (targetRb != null)
                {
                    targetRb.isKinematic = false;
                    targetRb.linearVelocity = Vector3.zero;
                    targetRb.angularVelocity = Vector3.zero;
                }

                EndEpisode();
                return;
            }
            else
            {
                AddReward(shapingPenalty);
                AddReward(Mathf.Clamp(progressDelta * currentCarryProgressScale, -0.05f, 0.05f));
                if (TrackStagnation(progressDelta))
                {
                    return;
                }
            }
        }

        if (target != null && target.position.y < -0.1f)
        {
            statsRecorder.Add("Custom/EndReason/OutOfBounds", 1f, StatAggregationMethod.Sum);
            SetReward(outOfBoundsPenalty);
            EndEpisode();
            return;
        }

        previousGoalDistance = GetDistanceToCurrentGoal();
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<float> c = actionsOut.ContinuousActions;
        c[0] = Input.GetAxis("Horizontal");
        c[1] = Input.GetAxis("Vertical");
        for (int i = 2; i < 6; i++)
        {
            c[i] = 0f;
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Table") || collision.gameObject.CompareTag("RobotLink"))
        {
            statsRecorder.Add("Custom/EndReason/SelfCollision", 1f, StatAggregationMethod.Sum);
            AddReward(-5f);
            EndEpisode();
        }
    }

    private void AcquireTarget()
    {
        if (partSpawner == null || partSpawner.CurrentPart == null)
        {
            return;
        }

        GameObject go = partSpawner.CurrentPart;
        target = go.transform;
        targetRb = go.GetComponent<Rigidbody>();
        targetCollider = go.GetComponent<Collider>();
        originalTargetParent = target.parent;
        targetHalfHeight = targetCollider != null ? targetCollider.bounds.extents.y : 0.02f;
        AttachObstacleForwarder(go);
    }

    private void AttachObstacleForwarders(Collider[] colliders)
    {
        if (colliders == null) return;
        foreach (Collider c in colliders)
        {
            if (c == null) continue;
            AttachObstacleForwarder(c.gameObject);
        }
    }

    private void CacheObstacles()
    {
        cachedObstacles = null;
        if (articulationRoot == null) return;
        Transform areaRoot = articulationRoot.transform.root;
        Collider[] all = areaRoot.GetComponentsInChildren<Collider>(true);
        List<Collider> list = new List<Collider>();
        for (int i = 0; i < all.Length; i++)
        {
            Collider c = all[i];
            if (c == null) continue;
            if (c.CompareTag(obstacleTag)) list.Add(c);
        }
        cachedObstacles = list.ToArray();
    }

    private float MinClearanceToObstacles(Vector3 worldPoint)
    {
        float minD = float.PositiveInfinity;
        if (cachedObstacles == null) return minD;
        for (int i = 0; i < cachedObstacles.Length; i++)
        {
            Collider c = cachedObstacles[i];
            if (c == null || !c.enabled) continue;
            Vector3 closest = c.ClosestPoint(worldPoint);
            float d = Vector3.Distance(worldPoint, closest);
            if (d < minD) minD = d;
        }
        return minD;
    }

    private void ApplyClearanceShaping()
    {
        if (clearanceDistance <= 0f || endEffector == null || cachedObstacles == null || cachedObstacles.Length == 0)
        {
            return;
        }

        float clearance = MinClearanceToObstacles(endEffector.position);
        if (statsRecorder != null && !float.IsPositiveInfinity(clearance))
        {
            statsRecorder.Add("Custom/MinClearance", clearance, StatAggregationMethod.Average);
        }

        if (clearance < clearanceDistance)
        {
            float t = 1f - (clearance / clearanceDistance);
            AddReward(-clearancePenaltyScale * t * t);
            if (statsRecorder != null)
            {
                statsRecorder.Add("Custom/ClearanceShapingHits", 1f, StatAggregationMethod.Sum);
            }
        }
    }

    private void AwardStateAdvance(int newState)
    {
        if (newState <= maxStateReached) return;
        maxStateReached = newState;
        if (newState == 4)
        {
            AddReward(carryToDropBonus);
            if (statsRecorder != null)
            {
                statsRecorder.Add("Custom/StateAdvance/CarryToDrop", 1f, StatAggregationMethod.Sum);
            }
        }
        else if (newState == 5)
        {
            AddReward(dropToReleaseBonus);
            if (statsRecorder != null)
            {
                statsRecorder.Add("Custom/StateAdvance/DropToRelease", 1f, StatAggregationMethod.Sum);
            }
        }
    }

    private void AttachObstacleForwarder(GameObject go)
    {
        if (go == null) return;
        ObstacleCollisionForwarder fwd = go.GetComponent<ObstacleCollisionForwarder>();
        if (fwd == null)
        {
            fwd = go.AddComponent<ObstacleCollisionForwarder>();
        }
        fwd.agent = this;
        fwd.targetTag = obstacleTag;
    }

    internal void ReportObstacleHit()
    {
        if (statsRecorder != null)
        {
            statsRecorder.Add("Custom/EndReason/ObstacleHit", 1f, StatAggregationMethod.Sum);
        }
        AddReward(obstacleHitPenalty);
        if (endEpisodeOnObstacleHit)
        {
            EndEpisode();
        }
    }

    private void RespawnAndAcquire()
    {
        if (partSpawner == null)
        {
            return;
        }

        partSpawner.SpawnPart();
        AcquireTarget();
    }

    private float GetDistanceToCurrentGoal()
    {
        EnsureReferencesResolved();

        if (endEffector == null)
        {
            return 0f;
        }

        return Vector3.Distance(endEffector.position, GetCurrentGoalPosition());
    }

    private bool TrackStagnation(float progressDelta)
    {
        if (progressDelta > progressEpsilon)
        {
            stagnationSteps = 0;
            return false;
        }

        stagnationSteps++;
        if (stagnationSteps < stagnationStepLimit)
        {
            return false;
        }

        statsRecorder.Add("Custom/EndReason/Stagnation", 1f, StatAggregationMethod.Sum);
        statsRecorder.Add("Custom/EndState", currentState, StatAggregationMethod.Average);
        SetReward(-0.5f);
        EndEpisode();
        return true;
    }

    private void EnsureReferencesResolved(bool force = false)
    {
        if (!force && referencesResolved && endEffector != null && dropZone != null)
        {
            return;
        }

        if (endEffector == null)
        {
            Transform flange = transform.root.Find("base/base_link/link_1/link_2/link_3/link_4/link_5/link_6/flange");
            if (flange != null)
            {
                endEffector = flange;
            }
        }

        if (gripAnchor == null && endEffector != null)
        {
            Transform tool0 = endEffector.Find("tool0");
            if (tool0 != null)
            {
                gripAnchor = tool0;
            }
        }

        if (dropZone == null)
        {
            GameObject dropZoneObject = GameObject.Find("drop-off");
            if (dropZoneObject != null)
            {
                dropZone = dropZoneObject.transform;
            }
        }

        referencesResolved = endEffector != null && dropZone != null;
    }

    private Vector3 GetDropZonePosition()
    {
        EnsureReferencesResolved();

        if (dropZone != null)
        {
            Collider dropCollider = dropZone.GetComponent<Collider>();
            if (dropCollider != null)
            {
                Bounds bounds = dropCollider.bounds;
                return new Vector3(bounds.center.x, bounds.max.y, bounds.center.z);
            }

            return dropZone.position;
        }

        return endEffector != null ? endEffector.position : Vector3.zero;
    }

    private Vector3 GetCurrentGoalPosition()
    {
        switch (currentState)
        {
            case 0:
                return GetPickupHoverPosition();
            case 1:
                return GetPickupHoverPosition();
            case 2:
                return GetPickupContactPosition();
            case 3:
                return GetDropHoverPosition();
            case 4:
                return GetDropHoverPosition();
            case 5:
                return GetDropReleasePosition();
            default:
                return endEffector != null ? endEffector.position : Vector3.zero;
        }
    }

    private Vector3 GetPickupHoverPosition()
    {
        if (target == null)
        {
            return endEffector != null ? endEffector.position : Vector3.zero;
        }

        return target.position + Vector3.up * (GetTargetHalfHeight() + pickupHoverHeight);
    }

    private Vector3 GetPickupContactPosition()
    {
        if (target == null)
        {
            return endEffector != null ? endEffector.position : Vector3.zero;
        }

        return target.position + Vector3.up * (GetTargetHalfHeight() + pickupContactClearance);
    }

    private Vector3 GetDropHoverPosition()
    {
        float targetHeight = GetTargetHalfHeight() * 2f;
        return GetDropZonePosition() + Vector3.up * (targetHeight + dropHoverHeight);
    }

    private Vector3 GetDropReleasePosition()
    {
        float targetHeight = GetTargetHalfHeight() * 2f;
        return GetDropZonePosition() + Vector3.up * (targetHeight + dropReleaseClearance);
    }

    private Vector3 GetHeldLocalPosition()
    {
        Transform anchor = gripAnchor != null ? gripAnchor : endEffector;
        if (anchor == null)
        {
            return Vector3.forward * (GetTargetHalfHeight() + heldPartClearance);
        }

        Vector3 worldApproachAxis;
        if (endEffector != null)
        {
            worldApproachAxis = endEffector.TransformDirection(flangeApproachAxis);
        }
        else
        {
            worldApproachAxis = anchor.TransformDirection(flangeApproachAxis);
        }

        Vector3 localApproachAxis = anchor.InverseTransformDirection(worldApproachAxis);
        localApproachAxis = localApproachAxis.sqrMagnitude > 0.0001f
            ? localApproachAxis.normalized
            : Vector3.forward;
        return localApproachAxis * (GetTargetHalfHeight() + heldPartClearance);
    }

    private float GetTargetHalfHeight()
    {
        return Mathf.Max(targetHalfHeight, 0.01f);
    }

    private void SetHeldCollisionIgnore(bool ignore)
    {
        if (holdingCollisionIgnored == ignore || targetCollider == null || robotColliders == null)
        {
            return;
        }

        foreach (Collider robotCollider in robotColliders)
        {
            if (robotCollider == null || robotCollider == targetCollider)
            {
                continue;
            }

            Physics.IgnoreCollision(targetCollider, robotCollider, ignore);
        }

        holdingCollisionIgnored = ignore;
    }

    private static void ReparentPreservingScale(Transform child, Transform newParent)
    {
        if (child == null) return;
        Vector3 worldScale = child.lossyScale;
        child.SetParent(newParent, true);
        if (newParent != null)
        {
            Vector3 parentScale = newParent.lossyScale;
            child.localScale = new Vector3(
                worldScale.x / parentScale.x,
                worldScale.y / parentScale.y,
                worldScale.z / parentScale.z);
        }
        else
        {
            child.localScale = worldScale;
        }
    }

    private static float GetHorizontalDistance(Vector3 a, Vector3 b)
    {
        a.y = 0f;
        b.y = 0f;
        return Vector3.Distance(a, b);
    }
}

public class ObstacleCollisionForwarder : MonoBehaviour
{
    public RobotArmAgent agent;
    public string targetTag = "Obstacle";

    private void OnCollisionEnter(Collision collision)
    {
        if (agent == null) return;
        if (!string.IsNullOrEmpty(targetTag) && collision.gameObject.CompareTag(targetTag))
        {
            agent.ReportObstacleHit();
        }
    }
}

