using Unity.MLAgents.Sensors;
using UnityEngine;

/// <summary>
/// Owns a top-down workspace camera + render texture that captures the whole pick-up/drop-off
/// area as the agent's vision input. The camera is disabled and only renders on demand
/// (one snapshot per episode), matching the project's "look at the world once, then act" spec.
/// </summary>
[RequireComponent(typeof(Camera))]
public class WorkspaceCameraRig : MonoBehaviour
{
    [Header("Render Target")]
    [Tooltip("Render texture written to by the workspace camera and read by the agent sensor.")]
    public RenderTexture renderTexture;
    [Tooltip("Resolution used to (re)create the render texture if none is assigned.")]
    public Vector2Int fallbackResolution = new Vector2Int(84, 84);

    [Header("Optional sensor wiring")]
    [Tooltip("If set, the rig will write the render texture into this sensor on Awake so it can be left blank in the inspector.")]
    public RenderTextureSensorComponent linkedSensor;

    private Camera workspaceCamera;
    private bool ownsRenderTexture;

    private void Awake()
    {
        workspaceCamera = GetComponent<Camera>();

        if (renderTexture == null)
        {
            renderTexture = new RenderTexture(
                fallbackResolution.x,
                fallbackResolution.y,
                16,
                RenderTextureFormat.ARGB32)
            {
                name = "WorkspaceCamera_RT",
                useMipMap = false,
                autoGenerateMips = false,
                wrapMode = TextureWrapMode.Clamp,
                filterMode = FilterMode.Bilinear,
            };
            renderTexture.Create();
            ownsRenderTexture = true;
        }

        workspaceCamera.targetTexture = renderTexture;
        // Camera renders on demand only; one Render() call per episode produces the snapshot.
        workspaceCamera.enabled = false;

        if (linkedSensor != null)
        {
            linkedSensor.RenderTexture = renderTexture;
        }
    }

    private void OnDestroy()
    {
        if (ownsRenderTexture && renderTexture != null)
        {
            renderTexture.Release();
            Destroy(renderTexture);
        }
    }

    /// <summary>
    /// Render the workspace into the render texture. Call once per episode at OnEpisodeBegin
    /// after the part has been respawned, so the snapshot reflects the new layout.
    /// </summary>
    public void CaptureSnapshot()
    {
        if (workspaceCamera == null || renderTexture == null)
        {
            return;
        }

        workspaceCamera.Render();
    }
}
