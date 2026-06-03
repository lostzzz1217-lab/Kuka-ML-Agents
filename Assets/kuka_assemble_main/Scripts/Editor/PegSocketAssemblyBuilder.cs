using UnityEditor;
using UnityEngine;
using System.Collections.Generic;

public static class PegSocketAssemblyBuilder
{
    private const string RootName = "PegAssemblyRoot";
    private const string GeneratedRootFolder = "Assets/Generated";
    private const string GeneratedBuilderFolder = "Assets/Generated/PegSocketBuilder";
    private const float PegRadius = 0.01f;
    private const float PegHeight = 0.06f;
    private const float SocketHoleRadialClearance = 0.0005f;
    private const float SocketHoleRadius = PegRadius + SocketHoleRadialClearance;
    private const float SocketOuterGuideRadius = SocketHoleRadius + 0.005f;
    private const float SocketBaseWidth = 0.08f;
    private const float SocketBaseDepth = 0.08f;
    private const float SocketBaseThickness = 0.02f;
    private const int SocketHoleSegmentCount = 32;

    [MenuItem("Tools/KUKA/Create Peg And Socket")]
    public static void CreatePegAndSocket()
    {
        var parent = Selection.activeTransform;
        var root = new GameObject(RootName);
        Undo.RegisterCreatedObjectUndo(root, "Create Peg And Socket");

        if (parent != null)
        {
            root.transform.SetParent(parent, false);
        }

        root.transform.position = parent != null ? parent.position : Vector3.zero;

        BuildWorkSurface(root.transform);
        BuildPeg(root.transform);
        BuildSocket(root.transform);

        Selection.activeGameObject = root;
        EditorGUIUtility.PingObject(root);
    }

    private static void BuildWorkSurface(Transform parent)
    {
        var table = GameObject.CreatePrimitive(PrimitiveType.Cube);
        Undo.RegisterCreatedObjectUndo(table, "Create Work Surface");
        table.name = "WorkSurface";
        table.transform.SetParent(parent, false);
        table.transform.localPosition = new Vector3(0f, -0.01f, 0f);
        table.transform.localScale = new Vector3(0.32f, 0.02f, 0.24f);

        var renderer = table.GetComponent<MeshRenderer>();
        renderer.sharedMaterial = CreateColoredMaterial("WorkSurface_Mat", new Color(0.32f, 0.34f, 0.38f, 1f));
    }

    private static void BuildPeg(Transform parent)
    {
        var pegRoot = new GameObject("Peg");
        Undo.RegisterCreatedObjectUndo(pegRoot, "Create Peg");
        pegRoot.transform.SetParent(parent, false);
        pegRoot.transform.localPosition = new Vector3(-0.07f, 0.03f, 0f);

        var pegVisual = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        Undo.RegisterCreatedObjectUndo(pegVisual, "Create Peg Visual");
        pegVisual.name = "PegVisual";
        pegVisual.transform.SetParent(pegRoot.transform, false);
        pegVisual.transform.localPosition = Vector3.zero;
        pegVisual.transform.localScale = new Vector3(PegRadius * 2f, PegHeight * 0.5f, PegRadius * 2f);

        var pegBody = pegRoot.AddComponent<Rigidbody>();
        pegBody.mass = 0.2f;
        pegBody.angularDamping = 0.2f;
        pegBody.linearDamping = 0.05f;
        pegBody.useGravity = true;

        var pegCollider = pegRoot.AddComponent<CapsuleCollider>();
        pegCollider.direction = 1;
        pegCollider.radius = PegRadius;
        pegCollider.height = PegHeight;
        pegCollider.center = Vector3.zero;

        var pegRenderer = pegVisual.GetComponent<MeshRenderer>();
        pegRenderer.sharedMaterial = CreateColoredMaterial("Peg_Mat", new Color(0.93f, 0.58f, 0.17f, 1f));

        var axis = new GameObject("PegAxis");
        Undo.RegisterCreatedObjectUndo(axis, "Create Peg Axis");
        axis.transform.SetParent(pegRoot.transform, false);
        axis.transform.localPosition = Vector3.zero;
        axis.transform.localRotation = Quaternion.identity;

        var attachPoint = new GameObject("PegAttachPoint");
        Undo.RegisterCreatedObjectUndo(attachPoint, "Create Peg Attach Point");
        attachPoint.transform.SetParent(pegRoot.transform, false);
        attachPoint.transform.localPosition = new Vector3(0f, PegHeight * 0.5f, 0f);
        attachPoint.transform.localRotation = Quaternion.identity;
    }

    private static void BuildSocket(Transform parent)
    {
        var socketRoot = new GameObject("Socket");
        Undo.RegisterCreatedObjectUndo(socketRoot, "Create Socket");
        socketRoot.transform.SetParent(parent, false);
        socketRoot.transform.localPosition = new Vector3(0.07f, 0f, 0f);

        var baseBlock = new GameObject("SocketBase");
        Undo.RegisterCreatedObjectUndo(baseBlock, "Create Socket Base");
        baseBlock.transform.SetParent(socketRoot.transform, false);
        baseBlock.transform.localPosition = new Vector3(0f, SocketBaseThickness * 0.5f, 0f);

        var meshFilter = baseBlock.AddComponent<MeshFilter>();
        var meshRenderer = baseBlock.AddComponent<MeshRenderer>();
        var meshCollider = baseBlock.AddComponent<MeshCollider>();
        var socketMesh = GetOrCreateSocketBaseMeshAsset(
            "SocketBase_Mesh",
            SocketBaseWidth,
            SocketBaseThickness,
            SocketBaseDepth,
            SocketHoleRadius,
            SocketHoleSegmentCount);

        meshFilter.sharedMesh = socketMesh;
        meshCollider.sharedMesh = socketMesh;
        meshRenderer.sharedMaterial =
            CreateColoredMaterial("SocketBase_Mat", new Color(0.18f, 0.45f, 0.78f, 1f));

        var preInsertPoint = new GameObject("PreInsertPoint");
        Undo.RegisterCreatedObjectUndo(preInsertPoint, "Create Pre Insert Point");
        preInsertPoint.transform.SetParent(socketRoot.transform, false);
        preInsertPoint.transform.localPosition = new Vector3(0f, 0.06f, 0f);

        var insertTarget = new GameObject("InsertTarget");
        Undo.RegisterCreatedObjectUndo(insertTarget, "Create Insert Target");
        insertTarget.transform.SetParent(socketRoot.transform, false);
        insertTarget.transform.localPosition = new Vector3(0f, 0.03f, 0f);

        var successZone = new GameObject("InsertSuccessZone");
        Undo.RegisterCreatedObjectUndo(successZone, "Create Insert Success Zone");
        successZone.transform.SetParent(socketRoot.transform, false);
        successZone.transform.localPosition = new Vector3(0f, 0.03f, 0f);

        var successCollider = successZone.AddComponent<BoxCollider>();
        successCollider.isTrigger = true;
        successCollider.size = new Vector3(SocketHoleRadius * 2f, 0.04f, SocketHoleRadius * 2f);

        var guideZone = new GameObject("SocketGuideZone");
        Undo.RegisterCreatedObjectUndo(guideZone, "Create Socket Guide Zone");
        guideZone.transform.SetParent(socketRoot.transform, false);
        guideZone.transform.localPosition = new Vector3(0f, 0.045f, 0f);

        var guideCollider = guideZone.AddComponent<BoxCollider>();
        guideCollider.isTrigger = true;
        guideCollider.size = new Vector3(SocketOuterGuideRadius * 2f, 0.03f, SocketOuterGuideRadius * 2f);

        var axis = new GameObject("SocketAxis");
        Undo.RegisterCreatedObjectUndo(axis, "Create Socket Axis");
        axis.transform.SetParent(socketRoot.transform, false);
        axis.transform.localPosition = Vector3.zero;
        axis.transform.localRotation = Quaternion.identity;
    }

    private static Mesh GetOrCreateSocketBaseMeshAsset(
        string assetName,
        float width,
        float height,
        float depth,
        float holeRadius,
        int segmentCount)
    {
        EnsureGeneratedFolders();

        var meshPath = $"{GeneratedBuilderFolder}/{assetName}.asset";
        var mesh = AssetDatabase.LoadAssetAtPath<Mesh>(meshPath);
        if (mesh == null)
        {
            mesh = new Mesh();
            mesh.name = assetName;
            AssetDatabase.CreateAsset(mesh, meshPath);
        }

        BuildSocketBaseMesh(mesh, width, height, depth, holeRadius, segmentCount);
        EditorUtility.SetDirty(mesh);
        AssetDatabase.SaveAssets();
        return mesh;
    }

    private static void BuildSocketBaseMesh(
        Mesh mesh,
        float width,
        float height,
        float depth,
        float holeRadius,
        int segmentCount)
    {
        var safeSegmentCount = Mathf.Max(8, Mathf.CeilToInt(segmentCount / 4f) * 4);
        var sideSegmentCount = safeSegmentCount / 4;
        var halfWidth = width * 0.5f;
        var halfHeight = height * 0.5f;
        var halfDepth = depth * 0.5f;

        var vertices = new List<Vector3>();
        var normals = new List<Vector3>();
        var uvs = new List<Vector2>();
        var triangles = new List<int>();

        var sideNormals = new[] { Vector3.right, Vector3.forward, Vector3.left, Vector3.back };

        for (var sideIndex = 0; sideIndex < 4; sideIndex++)
        {
            for (var segmentIndex = 0; segmentIndex < sideSegmentCount; segmentIndex++)
            {
                var topOuterA = GetSquareSidePoint(sideIndex, segmentIndex, sideSegmentCount, halfWidth, halfDepth, halfHeight);
                var topOuterB = GetSquareSidePoint(sideIndex, segmentIndex + 1, sideSegmentCount, halfWidth, halfDepth, halfHeight);
                var topInnerA = GetHoleArcPoint(sideIndex, segmentIndex, sideSegmentCount, holeRadius, halfHeight);
                var topInnerB = GetHoleArcPoint(sideIndex, segmentIndex + 1, sideSegmentCount, holeRadius, halfHeight);
                AddQuad(vertices, normals, uvs, triangles, topOuterA, topInnerA, topInnerB, topOuterB, Vector3.up);

                var bottomOuterA = GetSquareSidePoint(sideIndex, segmentIndex, sideSegmentCount, halfWidth, halfDepth, -halfHeight);
                var bottomOuterB = GetSquareSidePoint(sideIndex, segmentIndex + 1, sideSegmentCount, halfWidth, halfDepth, -halfHeight);
                var bottomInnerA = GetHoleArcPoint(sideIndex, segmentIndex, sideSegmentCount, holeRadius, -halfHeight);
                var bottomInnerB = GetHoleArcPoint(sideIndex, segmentIndex + 1, sideSegmentCount, holeRadius, -halfHeight);
                AddQuad(vertices, normals, uvs, triangles, bottomOuterA, bottomOuterB, bottomInnerB, bottomInnerA, Vector3.down);

                AddQuad(
                    vertices,
                    normals,
                    uvs,
                    triangles,
                    topOuterA,
                    bottomOuterA,
                    bottomOuterB,
                    topOuterB,
                    sideNormals[sideIndex]);
            }
        }

        for (var segmentIndex = 0; segmentIndex < safeSegmentCount; segmentIndex++)
        {
            var angleA = segmentIndex * Mathf.PI * 2f / safeSegmentCount;
            var angleB = (segmentIndex + 1) * Mathf.PI * 2f / safeSegmentCount;
            var topInnerA = new Vector3(Mathf.Cos(angleA) * holeRadius, halfHeight, Mathf.Sin(angleA) * holeRadius);
            var topInnerB = new Vector3(Mathf.Cos(angleB) * holeRadius, halfHeight, Mathf.Sin(angleB) * holeRadius);
            var bottomInnerA = new Vector3(Mathf.Cos(angleA) * holeRadius, -halfHeight, Mathf.Sin(angleA) * holeRadius);
            var bottomInnerB = new Vector3(Mathf.Cos(angleB) * holeRadius, -halfHeight, Mathf.Sin(angleB) * holeRadius);
            var midAngle = (angleA + angleB) * 0.5f;
            var inwardNormal = new Vector3(-Mathf.Cos(midAngle), 0f, -Mathf.Sin(midAngle));
            AddQuad(vertices, normals, uvs, triangles, topInnerA, topInnerB, bottomInnerB, bottomInnerA, inwardNormal);
        }

        mesh.Clear();
        mesh.name = "SocketBase_Mesh";
        mesh.SetVertices(vertices);
        mesh.SetNormals(normals);
        mesh.SetUVs(0, uvs);
        mesh.SetTriangles(triangles, 0);
        mesh.RecalculateBounds();
    }

    private static Vector3 GetSquareSidePoint(int sideIndex, int pointIndex, int sideSegmentCount, float halfWidth, float halfDepth, float y)
    {
        var t = pointIndex / (float)sideSegmentCount;
        return sideIndex switch
        {
            0 => new Vector3(halfWidth, y, Mathf.Lerp(-halfDepth, halfDepth, t)),
            1 => new Vector3(Mathf.Lerp(halfWidth, -halfWidth, t), y, halfDepth),
            2 => new Vector3(-halfWidth, y, Mathf.Lerp(halfDepth, -halfDepth, t)),
            _ => new Vector3(Mathf.Lerp(-halfWidth, halfWidth, t), y, -halfDepth),
        };
    }

    private static Vector3 GetHoleArcPoint(int sideIndex, int pointIndex, int sideSegmentCount, float radius, float y)
    {
        var angleDegrees = -45f + (sideIndex * 90f) + (pointIndex * 90f / sideSegmentCount);
        var angleRadians = angleDegrees * Mathf.Deg2Rad;
        return new Vector3(Mathf.Cos(angleRadians) * radius, y, Mathf.Sin(angleRadians) * radius);
    }

    private static void AddQuad(
        List<Vector3> vertices,
        List<Vector3> normals,
        List<Vector2> uvs,
        List<int> triangles,
        Vector3 a,
        Vector3 b,
        Vector3 c,
        Vector3 d,
        Vector3 expectedNormal)
    {
        var vertexStart = vertices.Count;
        var normal = expectedNormal.normalized;

        vertices.Add(a);
        vertices.Add(b);
        vertices.Add(c);
        vertices.Add(d);

        normals.Add(normal);
        normals.Add(normal);
        normals.Add(normal);
        normals.Add(normal);

        uvs.Add(new Vector2(a.x, a.z));
        uvs.Add(new Vector2(b.x, b.z));
        uvs.Add(new Vector2(c.x, c.z));
        uvs.Add(new Vector2(d.x, d.z));

        var facing = Vector3.Dot(Vector3.Cross(b - a, c - a), normal);
        if (facing >= 0f)
        {
            triangles.Add(vertexStart);
            triangles.Add(vertexStart + 1);
            triangles.Add(vertexStart + 2);
            triangles.Add(vertexStart);
            triangles.Add(vertexStart + 2);
            triangles.Add(vertexStart + 3);
            return;
        }

        triangles.Add(vertexStart);
        triangles.Add(vertexStart + 2);
        triangles.Add(vertexStart + 1);
        triangles.Add(vertexStart);
        triangles.Add(vertexStart + 3);
        triangles.Add(vertexStart + 2);
    }

    private static void EnsureGeneratedFolders()
    {
        if (!AssetDatabase.IsValidFolder(GeneratedRootFolder))
        {
            AssetDatabase.CreateFolder("Assets", "Generated");
        }

        if (!AssetDatabase.IsValidFolder(GeneratedBuilderFolder))
        {
            AssetDatabase.CreateFolder(GeneratedRootFolder, "PegSocketBuilder");
        }
    }

    private static Material CreateColoredMaterial(string assetName, Color color)
    {
        EnsureGeneratedFolders();

        var materialPath = $"{GeneratedBuilderFolder}/{assetName}.mat";
        var material = AssetDatabase.LoadAssetAtPath<Material>(materialPath);
        if (material != null)
        {
            return material;
        }

        var shader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
        material = new Material(shader);
        material.name = assetName;
        material.color = color;
        AssetDatabase.CreateAsset(material, materialPath);
        AssetDatabase.SaveAssets();
        return material;
    }
}
