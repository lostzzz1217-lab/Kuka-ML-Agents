using UnityEngine;

public class RandomPickupPartSpawner : MonoBehaviour
{
    public enum PartShape
    {
        Box,
        Cylinder
    }

    [Header("Spawn Area")]
    [SerializeField] private Collider pickUpArea;
    [SerializeField] private Transform spawnedRoot;
    [SerializeField] private bool autoSpawnOnStart = true;

    [Header("Part Settings")]
    [SerializeField] private PartShape partShape = PartShape.Box;
    [SerializeField] private Vector2 widthRange = new Vector2(0.05f, 0.08f);
    [SerializeField] private Vector2 depthRange = new Vector2(0.05f, 0.08f);
    [SerializeField] private Vector2 heightRange = new Vector2(0.03f, 0.05f);
    [SerializeField] private float edgePadding = 0.01f;
    [SerializeField] private float spawnLift = 0.002f;
    [SerializeField] private float partMass = 0.2f;
    [SerializeField] private Material partMaterial;
    [SerializeField] private string partTag = "Untagged";

    private GameObject currentPart;

    public GameObject CurrentPart => currentPart;

    private void Reset()
    {
        if (pickUpArea == null)
        {
            pickUpArea = GetComponent<Collider>();
        }
    }

    private void Start()
    {
        if (autoSpawnOnStart)
        {
            SpawnPart();
        }
    }

    [ContextMenu("Spawn Part")]
    public void SpawnPart()
    {
        if (pickUpArea == null)
        {
            Debug.LogError($"{nameof(RandomPickupPartSpawner)} requires a pick-up area collider.", this);
            return;
        }

        ClearPart();

        Vector3 partSize = GetRandomPartSize();
        Bounds areaBounds = pickUpArea.bounds;

        float halfX = partSize.x * 0.5f;
        float halfZ = partSize.z * 0.5f;

        float minX = areaBounds.min.x + halfX + edgePadding;
        float maxX = areaBounds.max.x - halfX - edgePadding;
        float minZ = areaBounds.min.z + halfZ + edgePadding;
        float maxZ = areaBounds.max.z - halfZ - edgePadding;

        if (minX > maxX || minZ > maxZ)
        {
            Debug.LogError("Pick-up area is too small for the configured part size and edge padding.", this);
            return;
        }

        Vector3 spawnPosition = new Vector3(
            Random.Range(minX, maxX),
            areaBounds.max.y + (partSize.y * 0.5f) + spawnLift,
            Random.Range(minZ, maxZ));

        PrimitiveType primitiveType = partShape == PartShape.Cylinder ? PrimitiveType.Cylinder : PrimitiveType.Cube;
        currentPart = GameObject.CreatePrimitive(primitiveType);
        currentPart.name = "TargetPart";
        currentPart.transform.SetPositionAndRotation(
            spawnPosition,
            Quaternion.Euler(0f, Random.Range(0f, 360f), 0f));
        currentPart.transform.localScale = partSize;

        Transform parent = EnsureSpawnedRoot();
        currentPart.transform.SetParent(parent, true);
        Vector3 parentScale = parent.lossyScale;
        currentPart.transform.localScale = new Vector3(
            partSize.x / parentScale.x,
            partSize.y / parentScale.y,
            partSize.z / parentScale.z);

        if (!string.IsNullOrWhiteSpace(partTag))
        {
            TryAssignTag(currentPart, partTag);
        }

        if (partMaterial != null)
        {
            Renderer renderer = currentPart.GetComponent<Renderer>();
            if (renderer != null)
            {
                renderer.sharedMaterial = partMaterial;
            }
        }

        Rigidbody rigidbody = currentPart.AddComponent<Rigidbody>();
        rigidbody.mass = partMass;
        rigidbody.angularDamping = 0.05f;
        rigidbody.linearDamping = 0.02f;
        rigidbody.interpolation = RigidbodyInterpolation.Interpolate;
        rigidbody.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
    }

    [ContextMenu("Clear Part")]
    public void ClearPart()
    {
        if (currentPart == null)
        {
            return;
        }

        if (Application.isPlaying)
        {
            Destroy(currentPart);
        }
        else
        {
            DestroyImmediate(currentPart);
        }

        currentPart = null;
    }

    private Vector3 GetRandomPartSize()
    {
        float width = Random.Range(widthRange.x, widthRange.y);
        float depth = Random.Range(depthRange.x, depthRange.y);
        float height = Random.Range(heightRange.x, heightRange.y);

        if (partShape == PartShape.Cylinder)
        {
            float diameter = Mathf.Min(width, depth);
            width = diameter;
            depth = diameter;
        }

        return new Vector3(width, height, depth);
    }

    private Transform EnsureSpawnedRoot()
    {
        if (spawnedRoot != null)
        {
            return spawnedRoot;
        }

        GameObject root = new GameObject("SpawnedParts");
        root.transform.SetParent(transform, false);
        spawnedRoot = root.transform;
        return spawnedRoot;
    }

    private static void TryAssignTag(GameObject target, string desiredTag)
    {
        try
        {
            target.tag = desiredTag;
        }
        catch (UnityException)
        {
            Debug.LogWarning($"Tag '{desiredTag}' does not exist. Keeping default tag on {target.name}.", target);
        }
    }
}
