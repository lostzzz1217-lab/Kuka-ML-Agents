using System;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

[DisallowMultipleComponent]
public class KukaAssembleAgent : Agent
{
    private enum TaskPhase
    {
        ApproachAndPick = 0,
        MoveToAssemblyHover = 1,
        HoldAboveSocket = 2,
        AlignAtAssembly = 3,
        DescendAndInsert = 4,
        ReleasePeg = 5,
        ReturnHome = 6,
    }

    [Header("Scene References")]
    [SerializeField] private Transform robotRoot;
    [SerializeField] private Transform endEffectorFrame;
    [SerializeField] private Transform gripAnchor;
    [SerializeField] private Camera observationCamera;
    [SerializeField] private Transform pegRoot;
    [SerializeField] private Rigidbody pegBody;
    [SerializeField] private Transform socketRoot;
    [SerializeField] private Transform preInsertPoint;
    [SerializeField] private Transform insertTarget;
    [SerializeField] private Transform socketAxis;
    [SerializeField] private Transform workSurface;

    [Header("Vision Snapshot")]
    [Tooltip("If true, peg/socket positions in observations come from a single rendered camera frame at episode start (color-segmented + ray-cast onto the work surface plane). Reward computation continues to use ground truth.")]
    [SerializeField] private bool useVisionSnapshot = true;
    [Tooltip("Camera used for the one-shot vision snapshot. Falls back to observationCamera if unset.")]
    [SerializeField] private Camera snapshotCamera;
    [SerializeField] private int snapshotResolution = 256;
    [Tooltip("Target color for peg pixels (RGB). Default tuned for the orange Peg_Mat.")]
    [SerializeField] private Color pegTargetColor = new Color(0.93f, 0.58f, 0.17f, 1f);
    [Tooltip("Target color for socket pixels (RGB). Default tuned for the blue SocketBase_Mat.")]
    [SerializeField] private Color socketTargetColor = new Color(0.18f, 0.45f, 0.78f, 1f);
    [Tooltip("Hue distance tolerance (0..1, where 1 = full color wheel). Match is on hue only — saturation/value gated by the thresholds below to ignore shadows and gray pixels. Robust to URP lighting since hue survives shading better than RGB.")]
    [SerializeField, Range(0.01f, 0.25f)] private float colorMatchTolerance = 0.06f;
    [Tooltip("Minimum saturation required for a pixel to be classified. Filters out gray/white work-surface pixels regardless of brightness.")]
    [SerializeField, Range(0.05f, 0.9f)] private float minSaturation = 0.25f;
    [Tooltip("Minimum value (brightness) required for a pixel to be classified. Filters out near-black shadows.")]
    [SerializeField, Range(0.0f, 0.5f)] private float minValue = 0.1f;
    [Tooltip("Minimum number of matching pixels required to consider a centroid valid.")]
    [SerializeField] private int minColorPixelCount = 4;
    [Tooltip("On vision-snapshot failure, save the rendered frame as PNG to <projectRoot>/visionDebug/ and log a hue histogram. Disable for training.")]
    [SerializeField] private bool debugSaveSnapshotOnFailure = false;

    [Header("Auto Discovery")]
    [SerializeField] private bool autoAssignReferences = true;

    [Header("Flange Orientation")]
    [SerializeField] private Vector3 flangeApproachAxis = Vector3.forward;
    [SerializeField] private float pickApproachAlignmentThreshold = 0.9f;
    [SerializeField] private float insertApproachAlignmentThreshold = 0.85f;
    [SerializeField] private float approachAlignmentRewardScale = 0.2f;

    [Header("Joint Control")]
    [SerializeField] private bool overrideJointDriveSettings = true;
    [SerializeField] private float jointDriveStiffness = 12000f;
    [SerializeField] private float jointDriveDamping = 1200f;
    [SerializeField] private float jointDriveForceLimit = 1500f;
    [SerializeField] private ArticulationDriveType jointDriveType = ArticulationDriveType.Target;
    [SerializeField] private float maxJointTargetDeltaDegrees = 2.5f;

    [Header("Episode")]
    [SerializeField] private int maxEnvironmentSteps = 2000;
    [SerializeField] private float workspaceRadius = 1.2f;
    [SerializeField] private float minPegHeight = -0.03f;
    [SerializeField] private bool randomizePegOnReset = true;
    [SerializeField] private Vector2 pegRandomOffsetX = new Vector2(-0.015f, 0.015f);
    [SerializeField] private Vector2 pegRandomOffsetZ = new Vector2(-0.015f, 0.015f);

    [Header("Pickup Geometry")]
    [SerializeField] private float pickupHoverHeight = 0.06f;
    [SerializeField] private float pickupContactClearance = 0.012f;
    [SerializeField] private float pickupHorizontalTolerance = 0.018f;
    [SerializeField] private float pickupVerticalTolerance = 0.02f;
    [SerializeField] private float pickupDistanceThreshold = 0.025f;
    [SerializeField] private float heldPartClearance = 0.0025f;

    [Header("Assembly Geometry")]
    [SerializeField] private float assemblyHoverHeight = 0.05f;
    [SerializeField] private float assemblyHoverDistanceThreshold = 0.03f;
    [SerializeField] private int socketHoldFramesRequired = 40;
    [Tooltip("Soft decay (frames per failed step) for the socket-hold counter so brief noise does not wipe hold progress. Mirrors assemblyAlignFrameDecay.")]
    [SerializeField] private int socketHoldFrameDecay = 3;
    [SerializeField] private float socketHoldHorizontalTolerance = 0.015f;
    [SerializeField] private float socketHoldVerticalTolerance = 0.015f;
    [SerializeField] private float assemblyRadialThreshold = 0.012f;
    [SerializeField] private float insertPositionThreshold = 0.01f;
    [SerializeField] private float pegAxisAlignmentThreshold = 0.95f;
    [SerializeField] private float endEffectorAbovePegTolerance = 0.005f;

    [Header("Release And Home")]
    [SerializeField] private int releaseSettleFramesRequired = 10;
    [SerializeField] private bool releaseUsesGravity = true;
    [SerializeField] private float releasedPegTolerance = 0.012f;
    [SerializeField] private float releaseVelocityTolerance = 0.025f;
    [SerializeField] private float homeJointToleranceDegrees = 2f;
    [SerializeField] private float homeEndEffectorTolerance = 0.03f;
    [SerializeField] private float homeRotationToleranceDegrees = 8f;
    [SerializeField] private float moveToAssemblyApproachSlack = 0.1f;
    [SerializeField] private float holdApproachSlack = 0.05f;

    [Header("Reward Weights")]
    [SerializeField] private float timePenalty = -0.0002f;
    [SerializeField] private float timeoutPenalty = -0.5f;
    [SerializeField] private float actionPenaltyScale = 0.0004f;
    [SerializeField] private float distanceProgressRewardScale = 2.0f;
    [SerializeField] private float alignmentProgressRewardScale = 0.3f;
    [SerializeField] private float insertionProgressRewardScale = 1.5f;
    [SerializeField] private float phaseTransitionReward = 1.0f;
    [SerializeField] private float hoverFrameReward = 0.01f;
    [SerializeField] private float graspReward = 4.0f;
    [SerializeField] private float releaseReward = 2.5f;
    [SerializeField] private float successReward = 20.0f;
    [SerializeField] private float failurePenalty = -2.0f;

    [Header("Align-Then-Insert Tuning")]
    [Tooltip("Frames the peg must satisfy radial+axis+approach alignment before the agent is allowed to commit to descent. Brief hold (not 100+) so policy can't bounce in/out of phase.")]
    [SerializeField] private int assemblyAlignHoldFramesRequired = 25;
    [Tooltip("Soft decay (frames per failed step) of the align-hold counter. Brief noise should not destroy hold progress.")]
    [SerializeField] private int assemblyAlignFrameDecay = 3;
    [Tooltip("Ratcheted bonus per insertion-depth quarter crossed (0%, 25%, 50%, 75%, 100%). Once banked it never erodes, so the agent is rewarded for committing to descent.")]
    [SerializeField] private float descendDepthBonusPerQuarter = 0.6f;

    [Header("Curriculum")]
    [Tooltip("Read 'curriculum_lesson' (0..2) from ML-Agents env params and lerp align thresholds, hold frames, and insertion shaping between an easy regime (0) and the serialized full-difficulty values (2).")]
    [SerializeField] private bool useCurriculum = true;
    [Tooltip("Lesson index used when ML-Agents env params are unavailable. 2 = full difficulty.")]
    [SerializeField] private float defaultLessonIndex = 2f;

    private struct CurriculumRuntime
    {
        public float lessonValue;
        public float pegAxisAlignmentThreshold;
        public float insertApproachAlignmentThreshold;
        public float assemblyRadialThreshold;
        public int assemblyAlignHoldFramesRequired;
        public float insertionProgressRewardScale;
        public bool requirePoseAlignmentForCompletion;
    }

    private CurriculumRuntime curriculum;

    private readonly List<ArticulationBody> controlledJoints = new();
    private readonly List<float> initialJointPositionsRadians = new();
    private readonly List<float> initialDriveTargetsRadians = new();
    private readonly List<float> zeroJointVelocities = new();
    private readonly List<float> initialJointTargetsDegrees = new();
    private readonly List<Collider> pegColliders = new();
    private readonly List<Collider> robotColliders = new();

    private ArticulationBody articulationRoot;
    private Transform initialPegParent;
    private Vector3 initialPegLocalPosition;
    private Quaternion initialPegLocalRotation;
    private Vector3 initialEndEffectorLocalPosition;
    private Quaternion initialEndEffectorLocalRotation;
    private TaskPhase currentPhase;
    private float previousPrimaryMetric;
    private float previousSecondaryMetric;
    private float previousTertiaryMetric;
    private bool pegAttached;
    private bool pegInserted;
    private bool episodeCompletingSuccessfully;
    private bool releasePerformed;
    private bool initialized;
    private float pegHalfExtent;
    private int holdStableFrames;
    private int assemblyAlignStableFrames;
    private int releaseStableFrames;
    private float maxInsertionDepth01;
    private int depthMilestonesCrossed;
    private int descendPhaseFrameCount;
    private bool episodeStatsRecorded;
    private Vector3 visionPegWorldPos;
    private Vector3 visionSocketWorldPos;
    private bool visionSnapshotValid;

    public int ControlledJointCount => controlledJoints.Count;

    public override void Initialize()
    {
        TryAutoAssignReferences();
        CacheSceneReferences();
        CacheControlledJoints();
        CacheInitialState();
        ConfigureJointDrives();
        ResetPhaseMetrics();
        RefreshCurriculum();
        if (maxEnvironmentSteps > 0)
        {
            MaxStep = maxEnvironmentSteps;
        }

        initialized = controlledJoints.Count > 0
            && pegRoot != null
            && insertTarget != null
            && endEffectorFrame != null;

        if (observationCamera == null)
        {
            Debug.LogWarning($"{nameof(KukaAssembleAgent)} did not find a root observation camera on {name}. Add a CameraSensorComponent to the global camera used for depth observations.", this);
        }

        if (!initialized)
        {
            Debug.LogWarning($"{nameof(KukaAssembleAgent)} is missing critical references on {name}.", this);
        }
    }

    public override void OnEpisodeBegin()
    {
        if (!initialized)
        {
            Initialize();
            if (!initialized)
            {
                return;
            }
        }

        RefreshCurriculum();
        ResetRobot();
        ResetPeg();

        pegAttached = false;
        pegInserted = false;
        episodeCompletingSuccessfully = false;
        releasePerformed = false;
        holdStableFrames = 0;
        assemblyAlignStableFrames = 0;
        releaseStableFrames = 0;
        maxInsertionDepth01 = 0f;
        depthMilestonesCrossed = 0;
        descendPhaseFrameCount = 0;
        episodeStatsRecorded = false;
        currentPhase = TaskPhase.ApproachAndPick;
        ResetPhaseMetrics();
        Academy.Instance.StatsRecorder.Add("Environment/Curriculum Lesson", curriculum.lessonValue, StatAggregationMethod.MostRecent);

        Physics.SyncTransforms();
        CaptureVisionSnapshot();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (!initialized)
        {
            sensor.AddObservation(new float[GetExpectedObservationSize()]);
            return;
        }

        for (var i = 0; i < controlledJoints.Count; i++)
        {
            var joint = controlledJoints[i];
            sensor.AddObservation(GetNormalizedJointPosition(joint));
            sensor.AddObservation(GetNormalizedJointVelocity(joint));
        }

        // Observation pipeline: peg/socket positions come from a one-shot vision snapshot taken
        // at OnEpisodeBegin (color segmentation + ray-cast onto work-surface plane). After the
        // virtual grasp, peg position is derived from end-effector pose (proprioception). Socket
        // is fixed, so its snapshot value is held for the episode. Reward computation continues
        // to use ground-truth transforms — only the policy's view of the world is restricted to
        // what a real depth camera + joint encoders could provide.
        var observedPegPos = GetObservedPegPosition();
        var observedSocketPos = GetObservedSocketPosition();
        var observedPegUp = GetObservedPegUp();
        var observedGoalPos = GetObservedGoalPosition();

        var localEndEffectorPosition = transform.InverseTransformPoint(endEffectorFrame.position) / Mathf.Max(workspaceRadius, 0.001f);
        var localEndEffectorForward = transform.InverseTransformDirection(endEffectorFrame.forward);
        var localEndEffectorUp = transform.InverseTransformDirection(endEffectorFrame.up);
        var localApproachDirection = transform.InverseTransformDirection(GetWorldApproachDirection());
        var localGoalDelta = transform.InverseTransformVector(observedGoalPos - endEffectorFrame.position) / Mathf.Max(workspaceRadius, 0.001f);
        var localPegToInsertDelta = transform.InverseTransformVector(observedSocketPos - observedPegPos) / Mathf.Max(workspaceRadius, 0.001f);
        var localPegUp = transform.InverseTransformDirection(observedPegUp);

        sensor.AddObservation(localEndEffectorPosition);
        sensor.AddObservation(localEndEffectorForward);
        sensor.AddObservation(localEndEffectorUp);
        sensor.AddObservation(localApproachDirection);
        sensor.AddObservation(localGoalDelta);
        sensor.AddObservation(Mathf.Clamp01(Vector3.Distance(endEffectorFrame.position, observedGoalPos) / Mathf.Max(workspaceRadius, 0.001f)));
        sensor.AddObservation(localPegToInsertDelta);
        sensor.AddObservation(Mathf.Clamp01(Vector3.Distance(observedPegPos, observedSocketPos) / Mathf.Max(workspaceRadius, 0.001f)));
        sensor.AddObservation(localPegUp);
        sensor.AddObservation(GetPickApproachAlignment());
        sensor.AddObservation(GetInsertApproachAlignment());
        sensor.AddObservation(GetObservedPegSocketAxisAlignment());
        var workspaceScale = Mathf.Max(workspaceRadius, 0.001f);
        sensor.AddObservation(Mathf.Clamp01(GetObservedPegRadialOffsetToSocketAxis() / workspaceScale));
        sensor.AddObservation(Mathf.Clamp(GetObservedPegAxialOffsetAlongSocketAxis() / workspaceScale, -1f, 1f));
        sensor.AddObservation(pegAttached ? 1f : 0f);
        sensor.AddObservation(pegInserted ? 1f : 0f);
        sensor.AddObservation(MaxStep > 0 ? Mathf.Clamp01((float)StepCount / MaxStep) : 0f);

        for (var i = 0; i < Enum.GetValues(typeof(TaskPhase)).Length; i++)
        {
            sensor.AddObservation(i == (int)currentPhase ? 1f : 0f);
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (!initialized)
        {
            return;
        }

        var scriptedPhase = currentPhase == TaskPhase.ReleasePeg;
        if (!scriptedPhase)
        {
            AddReward(timePenalty);
            ApplyJointActions(actions.ContinuousActions);
            AddReward(-GetActionPenalty(actions.ContinuousActions) * actionPenaltyScale);
        }

        UpdatePhaseRewardAndTransitions();
        ApplyTimeoutPenaltyIfNeeded();
        CheckTerminalConditions();
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActions = actionsOut.ContinuousActions;
        for (var i = 0; i < continuousActions.Length; i++)
        {
            continuousActions[i] = 0f;
        }
    }

    [ContextMenu("Auto Assign References")]
    private void AutoAssignReferencesFromContextMenu()
    {
        TryAutoAssignReferences();
    }

    private void Reset()
    {
        TryAutoAssignReferences();
    }

    private void TryAutoAssignReferences()
    {
        if (!autoAssignReferences)
        {
            return;
        }

        robotRoot ??= FindChildRecursive(transform, "kuka_kr6r700sixx");
        pegRoot ??= FindChildRecursive(transform, "Peg");
        pegBody ??= pegRoot != null ? pegRoot.GetComponent<Rigidbody>() : null;
        socketRoot ??= FindChildRecursive(transform, "Socket");
        preInsertPoint ??= FindChildRecursive(transform, "PreInsertPoint");
        insertTarget ??= FindChildRecursive(transform, "InsertTarget");
        socketAxis ??= FindChildRecursive(transform, "SocketAxis");
        endEffectorFrame ??= FindChildRecursive(transform, "flange") ?? FindChildRecursive(transform, "tool0");
        gripAnchor ??= FindChildRecursive(transform, "tool0") ?? endEffectorFrame;

        // Search root: the agent lives on kuka_kr6r700sixx (a child of the assemble prefab root),
        // so peg/socket/camera nodes that are siblings of kuka are NOT under transform. Use the
        // top-level prefab/scene root for these auto-bindings.
        var searchRoot = transform.root != null ? transform.root : transform;

        if (observationCamera == null)
        {
            var rootCamera = FindDirectChild(searchRoot, "Camera");
            if (rootCamera != null)
            {
                observationCamera = rootCamera.GetComponent<Camera>();
            }
        }

        if (snapshotCamera == null)
        {
            var snap = FindChildRecursive(searchRoot, "SnapshotCamera");
            if (snap != null)
            {
                snapshotCamera = snap.GetComponent<Camera>();
            }
        }

        // NOTE: use Unity-overloaded == here, not ??=. Serialized Object refs that point at a
        // since-destroyed/missing target read as "fake null" — passes C# null-coalescing but
        // fails Unity's null check. ??= would skip reassignment and we'd be stuck with a dead
        // reference. The explicit form correctly re-binds.
        if (workSurface == null)
        {
            workSurface = FindChildRecursive(searchRoot, "WorkSurface");
        }
    }

    private void CacheSceneReferences()
    {
        articulationRoot = robotRoot != null ? robotRoot.GetComponentInChildren<ArticulationBody>() : null;

        pegColliders.Clear();
        if (pegRoot != null)
        {
            pegColliders.AddRange(pegRoot.GetComponentsInChildren<Collider>(true));
            pegHalfExtent = ComputePegHalfExtent(Vector3.up);
        }

        robotColliders.Clear();
        if (robotRoot != null)
        {
            robotColliders.AddRange(robotRoot.GetComponentsInChildren<Collider>(true));
        }
    }

    private void CacheControlledJoints()
    {
        controlledJoints.Clear();

        if (robotRoot == null)
        {
            return;
        }

        var joints = robotRoot.GetComponentsInChildren<ArticulationBody>(true);
        Array.Sort(joints, (left, right) => left.index.CompareTo(right.index));

        foreach (var joint in joints)
        {
            if (joint.dofCount <= 0)
            {
                continue;
            }

            controlledJoints.Add(joint);
        }
    }

    private void CacheInitialState()
    {
        initialJointPositionsRadians.Clear();
        initialDriveTargetsRadians.Clear();
        initialJointTargetsDegrees.Clear();
        zeroJointVelocities.Clear();

        foreach (var joint in controlledJoints)
        {
            var drive = joint.xDrive;
            initialJointTargetsDegrees.Add(drive.target);
            initialDriveTargetsRadians.Add(drive.target * Mathf.Deg2Rad);
            initialJointPositionsRadians.Add(joint.dofCount > 0 ? joint.jointPosition[0] : 0f);
            zeroJointVelocities.Add(0f);
        }

        if (pegRoot != null)
        {
            initialPegParent = pegRoot.parent;
            initialPegLocalPosition = pegRoot.localPosition;
            initialPegLocalRotation = pegRoot.localRotation;
        }

        if (endEffectorFrame != null)
        {
            initialEndEffectorLocalPosition = transform.InverseTransformPoint(endEffectorFrame.position);
            initialEndEffectorLocalRotation = Quaternion.Inverse(transform.rotation) * endEffectorFrame.rotation;
        }
    }

    private void ConfigureJointDrives()
    {
        if (!overrideJointDriveSettings)
        {
            return;
        }

        foreach (var joint in controlledJoints)
        {
            var drive = joint.xDrive;
            drive.stiffness = jointDriveStiffness;
            drive.damping = jointDriveDamping;
            drive.forceLimit = jointDriveForceLimit;
            drive.driveType = jointDriveType;
            joint.xDrive = drive;
        }
    }

    private void ResetRobot()
    {
        for (var i = 0; i < controlledJoints.Count; i++)
        {
            var drive = controlledJoints[i].xDrive;
            drive.target = initialJointTargetsDegrees[i];
            drive.targetVelocity = 0f;
            controlledJoints[i].xDrive = drive;
        }

        if (articulationRoot != null && initialDriveTargetsRadians.Count == controlledJoints.Count)
        {
            articulationRoot.SetDriveTargets(initialDriveTargetsRadians);
            articulationRoot.SetJointPositions(initialJointPositionsRadians);
            articulationRoot.SetJointVelocities(zeroJointVelocities);
        }
    }

    private void ResetPeg()
    {
        if (pegRoot == null)
        {
            return;
        }

        ReparentPreservingScale(pegRoot, initialPegParent);
        pegRoot.localPosition = initialPegLocalPosition + GetRandomPegOffset();
        pegRoot.localRotation = initialPegLocalRotation;
        // Virtual grasp: keep robot↔peg collisions disabled across the entire ApproachAndPick
        // phase so the flange can sweep into grasp pose without knocking the peg over. Peg still
        // collides with WorkSurface (gravity + stand on table). Collision is only re-enabled in
        // ReleasePegIntoSocket so the dropped peg can interact with the socket.
        SetPegRobotCollisionIgnore(true);

        if (pegBody != null)
        {
            pegBody.isKinematic = false;
            pegBody.useGravity = true;
            pegBody.linearVelocity = Vector3.zero;
            pegBody.angularVelocity = Vector3.zero;
            pegBody.position = pegRoot.position;
            pegBody.rotation = pegRoot.rotation;
        }
    }

    private Vector3 GetRandomPegOffset()
    {
        if (!randomizePegOnReset)
        {
            return Vector3.zero;
        }

        return new Vector3(
            UnityEngine.Random.Range(pegRandomOffsetX.x, pegRandomOffsetX.y),
            0f,
            UnityEngine.Random.Range(pegRandomOffsetZ.x, pegRandomOffsetZ.y));
    }

    private void ApplyJointActions(ActionSegment<float> continuousActions)
    {
        var actionCount = Mathf.Min(continuousActions.Length, controlledJoints.Count);
        for (var i = 0; i < actionCount; i++)
        {
            var joint = controlledJoints[i];
            var drive = joint.xDrive;
            var delta = Mathf.Clamp(continuousActions[i], -1f, 1f) * maxJointTargetDeltaDegrees;
            drive.target = Mathf.Clamp(drive.target + delta, drive.lowerLimit, drive.upperLimit);
            joint.xDrive = drive;
        }
    }

    private void UpdatePhaseRewardAndTransitions()
    {
        switch (currentPhase)
        {
            case TaskPhase.ApproachAndPick:
                UpdateApproachAndPick();
                break;
            case TaskPhase.MoveToAssemblyHover:
                UpdateMoveToAssemblyHover();
                break;
            case TaskPhase.HoldAboveSocket:
                UpdateHoldAboveSocket();
                break;
            case TaskPhase.AlignAtAssembly:
                UpdateAlignAtAssembly();
                break;
            case TaskPhase.DescendAndInsert:
                UpdateDescendAndInsert();
                break;
            case TaskPhase.ReleasePeg:
                UpdateReleasePeg();
                break;
            case TaskPhase.ReturnHome:
                UpdateReturnHome();
                break;
        }
    }

    private void UpdateApproachAndPick()
    {
        var contactTarget = GetPickupContactPosition();
        var hoverTarget = GetPickupHoverPosition();
        var horizontalDistanceToPeg = GetPlanarDistance(endEffectorFrame.position, pegRoot.position);
        var currentTarget = horizontalDistanceToPeg <= pickupHorizontalTolerance * 1.5f
            ? contactTarget
            : hoverTarget;
        var distance = Vector3.Distance(endEffectorFrame.position, currentTarget);
        var verticalDistance = Mathf.Abs(endEffectorFrame.position.y - currentTarget.y);
        var approachAlignment = GetPickApproachAlignment();

        RewardMetricThatShouldDecrease(distance, distanceProgressRewardScale, ref previousPrimaryMetric);
        RewardMetricThatShouldIncrease(approachAlignment, approachAlignmentRewardScale, ref previousSecondaryMetric);

        var abovePeg = endEffectorFrame.position.y >= pegRoot.position.y - endEffectorAbovePegTolerance;
        if (horizontalDistanceToPeg <= pickupHorizontalTolerance
            && verticalDistance <= pickupVerticalTolerance
            && Vector3.Distance(endEffectorFrame.position, contactTarget) <= pickupDistanceThreshold
            && abovePeg
            && approachAlignment >= pickApproachAlignmentThreshold)
        {
            AttachPeg();
            AddReward(graspReward);
            AdvancePhase(TaskPhase.MoveToAssemblyHover, phaseTransitionReward);
        }
    }

    private void UpdateMoveToAssemblyHover()
    {
        var target = GetAssemblyHoverPosition();
        var distance = Vector3.Distance(endEffectorFrame.position, target);
        var approachAlignment = GetInsertApproachAlignment();

        RewardMetricThatShouldDecrease(distance, distanceProgressRewardScale, ref previousPrimaryMetric);
        RewardMetricThatShouldIncrease(approachAlignment, approachAlignmentRewardScale * 0.5f, ref previousSecondaryMetric);

        if (distance <= assemblyHoverDistanceThreshold
            && approachAlignment >= insertApproachAlignmentThreshold - moveToAssemblyApproachSlack)
        {
            AdvancePhase(TaskPhase.HoldAboveSocket, phaseTransitionReward);
        }
    }

    private void UpdateHoldAboveSocket()
    {
        var target = GetAssemblyHoverPosition();
        var horizontalDistance = GetPlanarDistance(endEffectorFrame.position, target);
        var verticalDistance = Mathf.Abs(endEffectorFrame.position.y - target.y);
        var approachAlignment = GetInsertApproachAlignment();

        RewardMetricThatShouldDecrease(horizontalDistance + verticalDistance, distanceProgressRewardScale, ref previousPrimaryMetric);
        RewardMetricThatShouldIncrease(approachAlignment, approachAlignmentRewardScale, ref previousSecondaryMetric);

        if (horizontalDistance <= socketHoldHorizontalTolerance
            && verticalDistance <= socketHoldVerticalTolerance
            && approachAlignment >= insertApproachAlignmentThreshold - holdApproachSlack)
        {
            holdStableFrames++;
            AddReward(hoverFrameReward);

            if (holdStableFrames >= socketHoldFramesRequired)
            {
                AdvancePhase(TaskPhase.AlignAtAssembly, phaseTransitionReward);
            }
        }
        else
        {
            holdStableFrames = Mathf.Max(0, holdStableFrames - socketHoldFrameDecay);
        }
    }

    private void UpdateAlignAtAssembly()
    {
        var radialOffset = GetPegRadialOffsetToSocketAxis();
        var axisAlignment = GetPegSocketAxisAlignment();
        var approachAlignment = GetInsertApproachAlignment();

        RewardMetricThatShouldDecrease(radialOffset, distanceProgressRewardScale, ref previousPrimaryMetric);
        RewardMetricThatShouldIncrease(axisAlignment, alignmentProgressRewardScale, ref previousSecondaryMetric);
        // Approach alignment was previously a static add — agent saw no gradient pushing it
        // toward the 0.85+ completion threshold. Shape it as progress so each step that
        // tilts the flange toward -socketAxis pays out.
        RewardMetricThatShouldIncrease(approachAlignment, approachAlignmentRewardScale, ref previousTertiaryMetric);

        if (radialOffset <= curriculum.assemblyRadialThreshold
            && axisAlignment >= curriculum.pegAxisAlignmentThreshold
            && approachAlignment >= curriculum.insertApproachAlignmentThreshold)
        {
            assemblyAlignStableFrames++;
            if (assemblyAlignStableFrames >= curriculum.assemblyAlignHoldFramesRequired)
            {
                AdvancePhase(TaskPhase.DescendAndInsert, phaseTransitionReward);
            }
        }
        else
        {
            assemblyAlignStableFrames = Mathf.Max(0, assemblyAlignStableFrames - assemblyAlignFrameDecay);
        }
    }

    private void UpdateDescendAndInsert()
    {
        descendPhaseFrameCount++;
        var pegToInsertDistance = Vector3.Distance(pegRoot.position, insertTarget.position);
        var insertionProgress = GetInsertionProgress01();
        var approachAlignment = GetInsertApproachAlignment();

        RewardMetricThatShouldDecrease(pegToInsertDistance, distanceProgressRewardScale, ref previousPrimaryMetric);
        RewardMetricThatShouldIncrease(insertionProgress, curriculum.insertionProgressRewardScale, ref previousSecondaryMetric);
        RewardMetricThatShouldIncrease(approachAlignment, approachAlignmentRewardScale * 0.5f, ref previousTertiaryMetric);

        // Ratcheted depth bonuses: once a quarter of insertion is reached it banks reward
        // permanently, so backsliding cannot erase it. This is the carrot that pushes the
        // agent to commit to descent rather than hover indefinitely "polishing" alignment.
        if (insertionProgress > maxInsertionDepth01)
        {
            maxInsertionDepth01 = insertionProgress;
            var milestone = Mathf.Clamp(Mathf.FloorToInt(insertionProgress * 4f), 0, 4);
            if (milestone > depthMilestonesCrossed)
            {
                AddReward(descendDepthBonusPerQuarter * (milestone - depthMilestonesCrossed));
                depthMilestonesCrossed = milestone;
            }
        }

        if (CanCompleteInsertion())
        {
            AddReward(releaseReward);
            AdvancePhase(TaskPhase.ReleasePeg, phaseTransitionReward);
        }
    }

    private void UpdateReleasePeg()
    {
        if (!releasePerformed)
        {
            ReleasePegIntoSocket();
            ResetPhaseMetrics();
        }

        var positionError = Vector3.Distance(pegRoot.position, insertTarget.position);
        var radialOffset = GetPegRadialOffsetToSocketAxis();
        var pegSpeed = pegBody != null ? pegBody.linearVelocity.magnitude : 0f;

        if (positionError <= releasedPegTolerance
            && radialOffset <= assemblyRadialThreshold
            && pegSpeed <= releaseVelocityTolerance)
        {
            releaseStableFrames++;

            if (releaseStableFrames >= releaseSettleFramesRequired)
            {
                FinalizeReleasedPeg();
                AdvancePhase(TaskPhase.ReturnHome, 0f);
            }
        }
        else
        {
            releaseStableFrames = 0;
        }
    }

    private void UpdateReturnHome()
    {
        var totalJointError = GetTotalHomeJointErrorDegrees();
        var endEffectorDistance = Vector3.Distance(endEffectorFrame.position, GetInitialEndEffectorWorldPosition());
        var endEffectorRotationError = Quaternion.Angle(endEffectorFrame.rotation, GetInitialEndEffectorWorldRotation());

        RewardMetricThatShouldDecrease(totalJointError, distanceProgressRewardScale * 0.15f, ref previousPrimaryMetric);
        RewardMetricThatShouldDecrease(endEffectorDistance + (endEffectorRotationError * 0.005f), distanceProgressRewardScale * 0.35f, ref previousSecondaryMetric);

        if (IsRobotHome())
        {
            episodeCompletingSuccessfully = true;
            AddReward(successReward);
            RecordEpisodeEndStats(0f);
            EndEpisode();
        }
    }

    private void AdvancePhase(TaskPhase nextPhase, float rewardBonus)
    {
        currentPhase = nextPhase;
        holdStableFrames = 0;
        assemblyAlignStableFrames = 0;
        releaseStableFrames = 0;
        if (nextPhase == TaskPhase.ReleasePeg)
        {
            releasePerformed = false;
        }

        AddReward(rewardBonus);
        ResetPhaseMetrics();
    }

    private void ResetPhaseMetrics()
    {
        previousPrimaryMetric = float.NaN;
        previousSecondaryMetric = float.NaN;
        previousTertiaryMetric = float.NaN;
    }

    private void RewardMetricThatShouldDecrease(float currentValue, float rewardScale, ref float previousValue)
    {
        if (float.IsNaN(previousValue))
        {
            previousValue = currentValue;
            return;
        }

        AddReward((previousValue - currentValue) * rewardScale);
        previousValue = currentValue;
    }

    private void RewardMetricThatShouldIncrease(float currentValue, float rewardScale, ref float previousValue)
    {
        if (float.IsNaN(previousValue))
        {
            previousValue = currentValue;
            return;
        }

        AddReward((currentValue - previousValue) * rewardScale);
        previousValue = currentValue;
    }

    private void AttachPeg()
    {
        pegAttached = true;
        pegInserted = false;

        var anchor = gripAnchor != null ? gripAnchor : endEffectorFrame;
        var desiredPegRotation = AlignUpAxis(pegRoot.rotation, -GetWorldApproachDirection());
        ReparentPreservingScale(pegRoot, anchor);
        pegRoot.position = anchor.TransformPoint(GetHeldLocalPosition(anchor));
        pegRoot.rotation = desiredPegRotation;

        if (pegBody != null)
        {
            pegBody.linearVelocity = Vector3.zero;
            pegBody.angularVelocity = Vector3.zero;
            pegBody.isKinematic = true;
            pegBody.useGravity = false;
        }

        SetPegRobotCollisionIgnore(true);
    }

    private void ReleasePegIntoSocket()
    {
        releasePerformed = true;
        pegAttached = false;
        pegInserted = true;

        if (socketRoot != null)
        {
            ReparentPreservingScale(pegRoot, socketRoot);
        }

        pegRoot.position = insertTarget.position;
        pegRoot.rotation = AlignUpAxis(pegRoot.rotation, GetSocketAxisWorld());
        SetPegRobotCollisionIgnore(false);

        if (pegBody != null)
        {
            pegBody.isKinematic = false;
            pegBody.useGravity = releaseUsesGravity;
            pegBody.position = pegRoot.position;
            pegBody.rotation = pegRoot.rotation;
            pegBody.linearVelocity = Vector3.zero;
            pegBody.angularVelocity = Vector3.zero;
        }
    }

    private void FinalizeReleasedPeg()
    {
        if (pegBody == null)
        {
            return;
        }

        pegBody.linearVelocity = Vector3.zero;
        pegBody.angularVelocity = Vector3.zero;
        pegBody.isKinematic = true;
        pegBody.useGravity = false;
        pegBody.position = pegRoot.position;
        pegBody.rotation = pegRoot.rotation;
    }

    private void ApplyTimeoutPenaltyIfNeeded()
    {
        if (episodeCompletingSuccessfully || MaxStep <= 0)
        {
            return;
        }

        if (StepCount < MaxStep - 1)
        {
            return;
        }

        AddReward(timeoutPenalty);
        RecordEpisodeEndStats(1f);
    }

    private void CheckTerminalConditions()
    {
        if (Vector3.Distance(pegRoot.position, transform.position) > workspaceRadius)
        {
            AddReward(failurePenalty);
            RecordEpisodeEndStats(2f);
            EndEpisode();
            return;
        }

        if (pegRoot.position.y < transform.position.y + minPegHeight)
        {
            AddReward(failurePenalty);
            RecordEpisodeEndStats(3f);
            EndEpisode();
            return;
        }

        if (!pegInserted && currentPhase >= TaskPhase.MoveToAssemblyHover && currentPhase < TaskPhase.ReleasePeg && !pegAttached)
        {
            AddReward(failurePenalty * 0.5f);
            RecordEpisodeEndStats(4f);
            EndEpisode();
        }
    }

    private void RecordEpisodeEndStats(float outcomeCode)
    {
        if (episodeStatsRecorded)
        {
            return;
        }
        episodeStatsRecorded = true;

        var stats = Academy.Instance.StatsRecorder;
        var approachAlignment = GetInsertApproachAlignment();
        var pegAxisAngleErrorDeg = Mathf.Acos(Mathf.Clamp(GetPegSocketAxisAlignment(), -1f, 1f)) * Mathf.Rad2Deg;
        var descendRatio = Mathf.Clamp01(maxInsertionDepth01);

        // Tags need the "Environment/" prefix to match the convention used by ML-Agents'
        // built-in stats (Cumulative Reward, Episode Length). Tensorboard panels that filter
        // by that prefix would otherwise miss these stats.
        stats.Add("Environment/Approach Alignment", approachAlignment, StatAggregationMethod.MostRecent);
        stats.Add("Environment/Final Peg Axis Angle Error", pegAxisAngleErrorDeg, StatAggregationMethod.MostRecent);
        stats.Add("Environment/Descend Phase Frames", descendPhaseFrameCount, StatAggregationMethod.MostRecent);
        stats.Add("Environment/Descend Progressing Ratio", descendRatio, StatAggregationMethod.MostRecent);
        stats.Add("Environment/Final Phase", (float)currentPhase, StatAggregationMethod.MostRecent);
        stats.Add("Environment/Episode Outcome", outcomeCode, StatAggregationMethod.MostRecent);
        stats.Add("Environment/Curriculum Lesson", curriculum.lessonValue, StatAggregationMethod.MostRecent);
    }

    private int GetExpectedObservationSize()
    {
        var jointCount = controlledJoints.Count > 0 ? controlledJoints.Count : 6;
        return jointCount * 2 + 38;
    }

    private Vector3 GetCurrentGoalPosition()
    {
        switch (currentPhase)
        {
            case TaskPhase.ApproachAndPick:
                return GetPlanarDistance(endEffectorFrame.position, pegRoot.position) <= pickupHorizontalTolerance * 1.5f
                    ? GetPickupContactPosition()
                    : GetPickupHoverPosition();
            case TaskPhase.MoveToAssemblyHover:
            case TaskPhase.HoldAboveSocket:
            case TaskPhase.AlignAtAssembly:
                return GetAssemblyHoverPosition();
            case TaskPhase.DescendAndInsert:
            case TaskPhase.ReleasePeg:
                return insertTarget != null ? insertTarget.position : transform.position;
            case TaskPhase.ReturnHome:
                return GetInitialEndEffectorWorldPosition();
            default:
                return transform.position;
        }
    }

    private Vector3 GetPickupHoverPosition()
    {
        return pegRoot != null
            ? pegRoot.position + transform.up * (Mathf.Max(pegHalfExtent, 0.01f) + pickupHoverHeight)
            : transform.position;
    }

    private Vector3 GetPickupContactPosition()
    {
        return pegRoot != null
            ? pegRoot.position + transform.up * (Mathf.Max(pegHalfExtent, 0.01f) + pickupContactClearance)
            : transform.position;
    }

    private Vector3 GetAssemblyHoverPosition()
    {
        if (preInsertPoint != null)
        {
            return preInsertPoint.position;
        }

        return insertTarget != null
            ? insertTarget.position + GetSocketAxisWorld() * (Mathf.Max(pegHalfExtent, 0.01f) + assemblyHoverHeight)
            : transform.position;
    }

    private float GetPegRadialOffsetToSocketAxis()
    {
        if (pegRoot == null || insertTarget == null)
        {
            return 0f;
        }

        var axis = GetSocketAxisWorld();
        var toPeg = pegRoot.position - insertTarget.position;
        var axialProjection = Vector3.Dot(toPeg, axis);
        var radialVector = toPeg - axis * axialProjection;
        return radialVector.magnitude;
    }

    // Signed distance along the socket axis from insertTarget to peg. Positive when the
    // peg sits on the "outside" of the socket (along +axis), zero at full insertion.
    private float GetPegAxialOffsetAlongSocketAxis()
    {
        if (pegRoot == null || insertTarget == null)
        {
            return 0f;
        }

        return Vector3.Dot(pegRoot.position - insertTarget.position, GetSocketAxisWorld());
    }

    private float GetPegSocketAxisAlignment()
    {
        if (pegRoot == null)
        {
            return 0f;
        }

        return Mathf.Abs(Vector3.Dot(pegRoot.up.normalized, GetSocketAxisWorld()));
    }

    private float GetInsertionProgress01()
    {
        if (pegRoot == null || insertTarget == null)
        {
            return 0f;
        }

        var start = preInsertPoint != null ? preInsertPoint.position : GetAssemblyHoverPosition();
        var end = insertTarget.position;
        var direction = end - start;
        var length = direction.magnitude;
        if (length <= 0.0001f)
        {
            return 0f;
        }

        var progress = Vector3.Dot(pegRoot.position - start, direction.normalized);
        return Mathf.Clamp01(progress / length);
    }

    private bool CanCompleteInsertion()
    {
        if (!pegAttached || pegRoot == null || insertTarget == null)
        {
            return false;
        }

        var radialOffset = GetPegRadialOffsetToSocketAxis();
        var positionError = Vector3.Distance(pegRoot.position, insertTarget.position);

        // Position is always required; radial uses curriculum-lerped threshold.
        if (positionError > insertPositionThreshold) return false;
        if (radialOffset > curriculum.assemblyRadialThreshold) return false;

        // Pose alignment is enforced from L1 onward. At L0 the snap-on-release
        // (ReleasePegIntoSocket) corrects orientation, so the agent can complete
        // the loop without yet mastering tight pose control.
        if (curriculum.requirePoseAlignmentForCompletion)
        {
            var axisAlignment = GetPegSocketAxisAlignment();
            var approachAlignment = GetInsertApproachAlignment();
            if (axisAlignment < curriculum.pegAxisAlignmentThreshold) return false;
            if (approachAlignment < curriculum.insertApproachAlignmentThreshold) return false;
        }

        return true;
    }

    private Vector3 GetSocketAxisWorld()
    {
        return socketAxis != null ? socketAxis.up.normalized : transform.up;
    }

    private Vector3 GetWorldApproachDirection()
    {
        var reference = endEffectorFrame != null ? endEffectorFrame : transform;
        var localAxis = flangeApproachAxis.sqrMagnitude > 0.0001f ? flangeApproachAxis.normalized : Vector3.forward;
        return reference.TransformDirection(localAxis).normalized;
    }

    private float GetPickApproachAlignment()
    {
        return Mathf.Clamp01(Vector3.Dot(GetWorldApproachDirection(), -transform.up));
    }

    private float GetInsertApproachAlignment()
    {
        return Mathf.Clamp01(Vector3.Dot(GetWorldApproachDirection(), -GetSocketAxisWorld()));
    }

    private Vector3 GetInitialEndEffectorWorldPosition()
    {
        return transform.TransformPoint(initialEndEffectorLocalPosition);
    }

    private Quaternion GetInitialEndEffectorWorldRotation()
    {
        return transform.rotation * initialEndEffectorLocalRotation;
    }

    private float GetTotalHomeJointErrorDegrees()
    {
        var totalError = 0f;
        for (var i = 0; i < controlledJoints.Count; i++)
        {
            if (controlledJoints[i].dofCount <= 0)
            {
                continue;
            }

            var currentDegrees = controlledJoints[i].jointPosition[0] * Mathf.Rad2Deg;
            var homeDegrees = initialJointPositionsRadians[i] * Mathf.Rad2Deg;
            totalError += Mathf.Abs(Mathf.DeltaAngle(currentDegrees, homeDegrees));
        }

        return totalError;
    }

    private bool IsRobotHome()
    {
        if (endEffectorFrame == null)
        {
            return true;
        }

        for (var i = 0; i < controlledJoints.Count; i++)
        {
            if (controlledJoints[i].dofCount <= 0)
            {
                continue;
            }

            var currentDegrees = controlledJoints[i].jointPosition[0] * Mathf.Rad2Deg;
            var homeDegrees = initialJointPositionsRadians[i] * Mathf.Rad2Deg;
            if (Mathf.Abs(Mathf.DeltaAngle(currentDegrees, homeDegrees)) > homeJointToleranceDegrees)
            {
                return false;
            }
        }

        var currentLocalPosition = transform.InverseTransformPoint(endEffectorFrame.position);
        if (Vector3.Distance(currentLocalPosition, initialEndEffectorLocalPosition) > homeEndEffectorTolerance)
        {
            return false;
        }

        var currentLocalRotation = Quaternion.Inverse(transform.rotation) * endEffectorFrame.rotation;
        return Quaternion.Angle(currentLocalRotation, initialEndEffectorLocalRotation) <= homeRotationToleranceDegrees;
    }

    private float GetNormalizedJointPosition(ArticulationBody joint)
    {
        if (joint.dofCount <= 0)
        {
            return 0f;
        }

        var drive = joint.xDrive;
        var lowerLimitRad = drive.lowerLimit * Mathf.Deg2Rad;
        var upperLimitRad = drive.upperLimit * Mathf.Deg2Rad;
        var position = joint.jointPosition[0];

        if (Mathf.Abs(upperLimitRad - lowerLimitRad) <= 0.0001f)
        {
            return Mathf.Clamp(position / Mathf.PI, -1f, 1f);
        }

        return Mathf.InverseLerp(lowerLimitRad, upperLimitRad, position) * 2f - 1f;
    }

    private float GetNormalizedJointVelocity(ArticulationBody joint)
    {
        if (joint.dofCount <= 0)
        {
            return 0f;
        }

        var maxVelocity = Mathf.Max(joint.maxJointVelocity, 0.001f);
        return Mathf.Clamp(joint.jointVelocity[0] / maxVelocity, -1f, 1f);
    }

    private Quaternion AlignUpAxis(Quaternion sourceRotation, Vector3 desiredUp)
    {
        var currentUp = sourceRotation * Vector3.up;
        if (currentUp.sqrMagnitude <= 0.0001f || desiredUp.sqrMagnitude <= 0.0001f)
        {
            return sourceRotation;
        }

        return Quaternion.FromToRotation(currentUp, desiredUp.normalized) * sourceRotation;
    }

    private Vector3 GetHeldLocalPosition(Transform anchor)
    {
        if (anchor == null)
        {
            return Vector3.zero;
        }

        var worldApproachDirection = GetWorldApproachDirection();
        var localApproachDirection = anchor.InverseTransformDirection(worldApproachDirection);
        if (localApproachDirection.sqrMagnitude <= 0.0001f)
        {
            localApproachDirection = Vector3.forward;
        }

        localApproachDirection.Normalize();
        return localApproachDirection * (Mathf.Max(pegHalfExtent, 0.01f) + heldPartClearance);
    }

    private float ComputePegHalfExtent(Vector3 worldDirection)
    {
        if (pegColliders.Count == 0)
        {
            return 0.02f;
        }

        var direction = worldDirection.sqrMagnitude > 0.0001f ? worldDirection.normalized : Vector3.up;
        var maxProjectedExtent = 0.02f;

        foreach (var collider in pegColliders)
        {
            if (collider == null)
            {
                continue;
            }

            var extents = collider.bounds.extents;
            var projectedExtent =
                Mathf.Abs(direction.x) * extents.x +
                Mathf.Abs(direction.y) * extents.y +
                Mathf.Abs(direction.z) * extents.z;
            maxProjectedExtent = Mathf.Max(maxProjectedExtent, projectedExtent);
        }

        return maxProjectedExtent;
    }

    private void SetPegRobotCollisionIgnore(bool ignore)
    {
        if (pegColliders.Count == 0 || robotColliders.Count == 0)
        {
            return;
        }

        foreach (var pegCollider in pegColliders)
        {
            if (pegCollider == null)
            {
                continue;
            }

            foreach (var robotCollider in robotColliders)
            {
                if (robotCollider == null || pegCollider == robotCollider)
                {
                    continue;
                }

                Physics.IgnoreCollision(pegCollider, robotCollider, ignore);
            }
        }
    }

    private static void ReparentPreservingScale(Transform child, Transform newParent)
    {
        if (child == null)
        {
            return;
        }

        var worldScale = child.lossyScale;
        child.SetParent(newParent, true);

        if (newParent == null)
        {
            child.localScale = worldScale;
            return;
        }

        var parentScale = newParent.lossyScale;
        child.localScale = new Vector3(
            parentScale.x == 0f ? worldScale.x : worldScale.x / parentScale.x,
            parentScale.y == 0f ? worldScale.y : worldScale.y / parentScale.y,
            parentScale.z == 0f ? worldScale.z : worldScale.z / parentScale.z);
    }

    private static Transform FindDirectChild(Transform parent, string childName)
    {
        if (parent == null)
        {
            return null;
        }

        for (var i = 0; i < parent.childCount; i++)
        {
            var child = parent.GetChild(i);
            if (child.name == childName)
            {
                return child;
            }
        }

        return null;
    }

    private static Transform FindChildRecursive(Transform parent, string childName)
    {
        if (parent == null)
        {
            return null;
        }

        var children = parent.GetComponentsInChildren<Transform>(true);
        foreach (var child in children)
        {
            if (child.name == childName)
            {
                return child;
            }
        }

        return null;
    }

    private float GetPlanarDistance(Vector3 a, Vector3 b)
    {
        return Vector3.ProjectOnPlane(a - b, transform.up).magnitude;
    }

    private void RefreshCurriculum()
    {
        // Sensible fallback depends on context:
        // - Trainer connected (training): an env-param-not-yet-pushed condition is just
        //   the startup race window. Default to L0 (0f) so we don't briefly run agents
        //   at full difficulty before the first curriculum value arrives. Without this
        //   guard a build-mode training run starts every agent at lesson=defaultLessonIndex
        //   (2 = L2), which crushes initial reward and prevents reward-gated curriculum
        //   from ever advancing — observed on the 0508_fresh run.
        // - No trainer (inference / Heuristic): use the serialized defaultLessonIndex,
        //   typically 2.0, since we want to evaluate the deployed model at full difficulty.
        var fallback = Academy.Instance.IsCommunicatorOn ? 0f : defaultLessonIndex;
        var lesson = fallback;
        if (useCurriculum)
        {
            try
            {
                var envParams = Academy.Instance.EnvironmentParameters;
                lesson = envParams.GetWithDefault("curriculum_lesson", fallback);
            }
            catch (Exception)
            {
                lesson = fallback;
            }
        }

        lesson = Mathf.Clamp(lesson, 0f, 2f);
        if (!Mathf.Approximately(lesson, curriculum.lessonValue))
        {
            Debug.Log($"[KukaAssembleAgent] curriculum_lesson = {lesson:F2} (was {curriculum.lessonValue:F2})", this);
        }
        // Piecewise softer ramp. Old mapping was linear (lesson=1.0 → t=0.5), which made the
        // L0 → L1 step too steep — policy collapsed in the 0507_vision_v1 run as soon as L1
        // started enforcing pose alignment with mid thresholds. Now lesson=1.0 → t=0.25, so
        // L1 is closer to L0 than to L2 (gentle alignment phase-in). L2 still hits t=1.0.
        var t = lesson <= 1f
            ? lesson * 0.25f                  // L0→L1: 0 → 0.25 (was 0 → 0.5)
            : 0.25f + (lesson - 1f) * 0.75f;  // L1→L2: 0.25 → 1.0 (steeper, but agent has L1 grounding)

        // L0 (t=0):    loose alignment, brief hold, position-only completion, boosted insertion shaping.
        // L1 (t=0.25): mostly L0-like thresholds + full pose required for completion (the gating step).
        // L2 (t=1.0):  serialized full difficulty. Production-grade alignment + insertion.
        curriculum = new CurriculumRuntime
        {
            lessonValue = lesson,
            pegAxisAlignmentThreshold = Mathf.Lerp(0.5f, pegAxisAlignmentThreshold, t),
            insertApproachAlignmentThreshold = Mathf.Lerp(0.5f, insertApproachAlignmentThreshold, t),
            assemblyRadialThreshold = Mathf.Lerp(0.04f, assemblyRadialThreshold, t),
            assemblyAlignHoldFramesRequired = Mathf.RoundToInt(Mathf.Lerp(5f, assemblyAlignHoldFramesRequired, t)),
            insertionProgressRewardScale = Mathf.Lerp(insertionProgressRewardScale * 2f, insertionProgressRewardScale, t),
            requirePoseAlignmentForCompletion = lesson > 0.5f,
        };
    }

    private static float GetActionPenalty(ActionSegment<float> continuousActions)
    {
        var penalty = 0f;
        for (var i = 0; i < continuousActions.Length; i++)
        {
            penalty += continuousActions[i] * continuousActions[i];
        }

        return penalty;
    }

    // -------------------- Vision snapshot --------------------
    // Renders a single frame at episode start, color-segments peg + socket, and
    // ray-casts the pixel centroids onto the work-surface plane to recover world
    // positions. Replaces ground-truth peg/socket positions in observations only.
    private void CaptureVisionSnapshot()
    {
        visionSnapshotValid = false;
        if (!useVisionSnapshot)
        {
            return;
        }

        var cam = snapshotCamera != null ? snapshotCamera : observationCamera;
        if (cam == null || pegRoot == null || (insertTarget == null && socketRoot == null))
        {
            return;
        }

        var resolution = Mathf.Max(32, snapshotResolution);
        RenderTexture rt = null;
        Texture2D tex = null;
        var prevTarget = cam.targetTexture;
        var prevActive = RenderTexture.active;

        try
        {
            rt = RenderTexture.GetTemporary(resolution, resolution, 16, RenderTextureFormat.ARGB32);
            cam.targetTexture = rt;
            cam.Render();

            RenderTexture.active = rt;
            tex = new Texture2D(resolution, resolution, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, resolution, resolution), 0, 0);
            tex.Apply(false);

            var pixels = tex.GetPixels32();
            var pegCount = ComputeColorCentroid(pixels, resolution, pegTargetColor, colorMatchTolerance, out var pegCentroidPx);
            var socketCount = ComputeColorCentroid(pixels, resolution, socketTargetColor, colorMatchTolerance, out var socketCentroidPx);

            var planeY = workSurface != null ? workSurface.position.y : transform.position.y;
            var plane = new Plane(Vector3.up, new Vector3(0f, planeY, 0f));

            var pegWorld = Vector3.zero;
            var socketWorld = Vector3.zero;
            var pegOk = pegCount >= minColorPixelCount
                && TryProjectViewportToPlane(cam, pegCentroidPx / resolution, plane, out pegWorld);
            var socketOk = socketCount >= minColorPixelCount
                && TryProjectViewportToPlane(cam, socketCentroidPx / resolution, plane, out socketWorld);

            if (pegOk)
            {
                visionPegWorldPos = pegWorld;
            }
            if (socketOk)
            {
                visionSocketWorldPos = socketWorld;
            }

            visionSnapshotValid = pegOk && socketOk;

            if (!visionSnapshotValid)
            {
                Debug.LogWarning(
                    $"[KukaAssembleAgent] vision snapshot incomplete: pegPx={pegCount}, socketPx={socketCount}. Falling back to ground truth this episode.",
                    this);
                if (debugSaveSnapshotOnFailure)
                {
                    DumpSnapshotForDebug(tex, pixels);
                }
                visionPegWorldPos = pegRoot.position;
                visionSocketWorldPos = insertTarget != null ? insertTarget.position : socketRoot.position;
            }
        }
        finally
        {
            cam.targetTexture = prevTarget;
            RenderTexture.active = prevActive;
            if (rt != null)
            {
                RenderTexture.ReleaseTemporary(rt);
            }
            if (tex != null)
            {
                Destroy(tex);
            }
        }
    }

    private int ComputeColorCentroid(Color32[] pixels, int size, Color target, float hueTolerance, out Vector2 centroid)
    {
        Color.RGBToHSV(target, out var targetH, out _, out _);
        var minSat = minSaturation;
        var minVal = minValue;

        long sumX = 0, sumY = 0;
        var count = 0;
        for (var y = 0; y < size; y++)
        {
            var row = y * size;
            for (var x = 0; x < size; x++)
            {
                var c = pixels[row + x];
                Color.RGBToHSV(new Color(c.r * (1f / 255f), c.g * (1f / 255f), c.b * (1f / 255f)), out var h, out var s, out var v);
                if (s < minSat || v < minVal)
                {
                    continue;
                }

                var hueDist = Mathf.Abs(h - targetH);
                if (hueDist > 0.5f)
                {
                    hueDist = 1f - hueDist; // wrap around the color wheel
                }
                if (hueDist < hueTolerance)
                {
                    sumX += x;
                    sumY += y;
                    count++;
                }
            }
        }

        centroid = count > 0 ? new Vector2((float)sumX / count, (float)sumY / count) : Vector2.zero;
        return count;
    }

    private void DumpSnapshotForDebug(Texture2D tex, Color32[] pixels)
    {
        try
        {
            var dir = System.IO.Path.Combine(Application.dataPath, "..", "visionDebug");
            System.IO.Directory.CreateDirectory(dir);
            var stamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss_fff");
            var pngPath = System.IO.Path.Combine(dir, $"snapshot_{stamp}.png");
            System.IO.File.WriteAllBytes(pngPath, tex.EncodeToPNG());

            // 12-bin hue histogram on saturated pixels (sat >= 0.15) — coarse but enough to
            // see whether peg-orange pixels exist at all and where they cluster.
            var bins = new int[12];
            var saturatedTotal = 0;
            foreach (var c in pixels)
            {
                Color.RGBToHSV(new Color(c.r * (1f / 255f), c.g * (1f / 255f), c.b * (1f / 255f)), out var h, out var s, out var v);
                if (s < 0.15f || v < 0.05f)
                {
                    continue;
                }
                saturatedTotal++;
                var idx = Mathf.Clamp(Mathf.FloorToInt(h * 12f), 0, 11);
                bins[idx]++;
            }
            var hist = string.Join(",", bins);
            Debug.LogWarning($"[KukaAssembleAgent] saved {pngPath}; saturated={saturatedTotal}; hue-bins(12)=[{hist}]; peg-target-bin={Mathf.FloorToInt(0.0899f * 12f)}; socket-target-bin={Mathf.FloorToInt(0.5917f * 12f)}", this);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[KukaAssembleAgent] vision debug dump failed: {e.Message}", this);
        }
    }

    private static bool TryProjectViewportToPlane(Camera cam, Vector2 viewportXY, Plane plane, out Vector3 world)
    {
        var ray = cam.ViewportPointToRay(new Vector3(viewportXY.x, viewportXY.y, 0f));
        if (plane.Raycast(ray, out var dist))
        {
            world = ray.GetPoint(dist);
            return true;
        }
        world = Vector3.zero;
        return false;
    }

    // -------------------- Observation-only views --------------------
    // These shadow the ground-truth helpers and are used only inside CollectObservations.
    private Vector3 GetObservedPegPosition()
    {
        if (!useVisionSnapshot || !visionSnapshotValid)
        {
            return pegRoot != null ? pegRoot.position : transform.position;
        }

        if (pegAttached)
        {
            // After virtual grasp the peg moves rigidly with the gripper. Real-world parity:
            // the policy can derive this from joint encoders + grip offset, no vision needed.
            var anchor = gripAnchor != null ? gripAnchor : endEffectorFrame;
            if (anchor != null)
            {
                return anchor.TransformPoint(GetHeldLocalPosition(anchor));
            }
        }

        return visionPegWorldPos;
    }

    private Vector3 GetObservedSocketPosition()
    {
        if (!useVisionSnapshot || !visionSnapshotValid)
        {
            return insertTarget != null
                ? insertTarget.position
                : (socketRoot != null ? socketRoot.position : transform.position);
        }
        return visionSocketWorldPos;
    }

    private Vector3 GetObservedPegUp()
    {
        if (pegAttached)
        {
            // Held peg is aligned by AttachPeg so its up axis = -approach direction.
            return -GetWorldApproachDirection();
        }
        // Free-standing peg sits upright on the work surface — derivable from the calibrated
        // table normal, not from per-frame ground truth.
        return Vector3.up;
    }

    private Vector3 GetObservedGoalPosition()
    {
        switch (currentPhase)
        {
            case TaskPhase.ApproachAndPick:
            {
                var pegPos = GetObservedPegPosition();
                var planar = GetPlanarDistance(endEffectorFrame.position, pegPos);
                var hoverHeight = Mathf.Max(pegHalfExtent, 0.01f) + pickupHoverHeight;
                var contactHeight = Mathf.Max(pegHalfExtent, 0.01f) + pickupContactClearance;
                var height = planar <= pickupHorizontalTolerance * 1.5f ? contactHeight : hoverHeight;
                return pegPos + transform.up * height;
            }
            case TaskPhase.MoveToAssemblyHover:
            case TaskPhase.HoldAboveSocket:
            case TaskPhase.AlignAtAssembly:
            {
                var socketPos = GetObservedSocketPosition();
                return socketPos + GetSocketAxisWorld() * (Mathf.Max(pegHalfExtent, 0.01f) + assemblyHoverHeight);
            }
            case TaskPhase.DescendAndInsert:
            case TaskPhase.ReleasePeg:
                return GetObservedSocketPosition();
            case TaskPhase.ReturnHome:
                return GetInitialEndEffectorWorldPosition();
            default:
                return transform.position;
        }
    }

    private float GetObservedPegRadialOffsetToSocketAxis()
    {
        var axis = GetSocketAxisWorld();
        var toPeg = GetObservedPegPosition() - GetObservedSocketPosition();
        var axialProjection = Vector3.Dot(toPeg, axis);
        var radialVector = toPeg - axis * axialProjection;
        return radialVector.magnitude;
    }

    private float GetObservedPegAxialOffsetAlongSocketAxis()
    {
        return Vector3.Dot(GetObservedPegPosition() - GetObservedSocketPosition(), GetSocketAxisWorld());
    }

    private float GetObservedPegSocketAxisAlignment()
    {
        return Mathf.Abs(Vector3.Dot(GetObservedPegUp().normalized, GetSocketAxisWorld()));
    }
}
