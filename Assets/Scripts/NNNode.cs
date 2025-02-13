using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using UnityEngine.UIElements;

public class NNNode
{
    public float value;
    public float sum;
    public float error;

    public List<NNEdge> edgesIn = new List<NNEdge>();
    public List<NNEdge> edgesOut = new List<NNEdge>();
}
