using UnityEditor;
using UnityEngine;

public static class SnapshotCameraSetup
{
    [MenuItem("Tools/KUKA/Add Or Reset SnapshotCamera On Selected TrainingArea")]
    public static void AddOrResetSnapshotCamera()
    {
        var selection = Selection.activeGameObject;
        if (selection == null)
        {
            EditorUtility.DisplayDialog("SnapshotCamera", "Select the TrainingArea GameObject (the one carrying KukaAssembleAgent), then run the menu again.", "OK");
            return;
        }

        var trainingArea = selection.transform;
        var assemblyRoot = FindChildByName(trainingArea, "PegAssemblyRoot")
                          ?? FindChildByName(trainingArea, "Socket")
                          ?? FindChildByName(trainingArea, "WorkSurface");
        if (assemblyRoot == null)
        {
            EditorUtility.DisplayDialog("SnapshotCamera", "Could not find PegAssemblyRoot / Socket / WorkSurface under the selected GameObject. Select TrainingArea and try again.", "OK");
            return;
        }

        var existing = FindChildByName(trainingArea, "SnapshotCamera");
        GameObject camGo;
        if (existing != null)
        {
            camGo = existing.gameObject;
            Undo.RecordObject(camGo.transform, "Reset SnapshotCamera");
        }
        else
        {
            camGo = new GameObject("SnapshotCamera");
            Undo.RegisterCreatedObjectUndo(camGo, "Create SnapshotCamera");
            camGo.transform.SetParent(trainingArea, false);
        }

        var cam = camGo.GetComponent<Camera>();
        if (cam == null)
        {
            cam = Undo.AddComponent<Camera>(camGo);
        }

        // Position 0.6m straight above the assembly root, looking straight down.
        var assemblyWorld = assemblyRoot.position;
        var camWorldPos = assemblyWorld + Vector3.up * 0.6f;
        camGo.transform.position = camWorldPos;
        camGo.transform.rotation = Quaternion.LookRotation(Vector3.down, Vector3.forward);

        cam.orthographic = true;
        cam.orthographicSize = 0.18f;            // ~36cm field of view, comfortably covers a 12cm peg-socket spread
        cam.nearClipPlane = 0.05f;
        cam.farClipPlane = 2.0f;
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = new Color(0.5f, 0.5f, 0.5f, 1f);
        cam.allowHDR = false;                    // disables HDR/tonemapping which can desaturate bright orange
        cam.allowMSAA = false;
        cam.targetDisplay = 7;                   // route off any visible Game View

        EditorUtility.SetDirty(camGo);
        Selection.activeGameObject = camGo;
        Debug.Log($"[SnapshotCameraSetup] Configured SnapshotCamera at {camWorldPos:F3} looking down at {assemblyWorld:F3}.", camGo);
    }

    private static Transform FindChildByName(Transform parent, string n)
    {
        if (parent == null)
        {
            return null;
        }
        var all = parent.GetComponentsInChildren<Transform>(true);
        foreach (var t in all)
        {
            if (t.name == n)
            {
                return t;
            }
        }
        return null;
    }
}
