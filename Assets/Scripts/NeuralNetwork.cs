using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;

public class NeuralNetwork
{
    public int layers;
    public NNNode[][] nodes;
    public List<NNEdge> edges = new List<NNEdge>();

    public NeuralNetwork(int numLayers, int hlLength)
    {
        //first, set up each layer
        layers = numLayers;

        nodes = new NNNode[layers][];

        //input layer
        nodes[0] = new NNNode[64];

        //hidden layers
        for (int i = 1; i < layers - 1; i++)
        {
            nodes[i] = new NNNode[hlLength];
        }

        //output layer
        nodes[layers-1] = new NNNode[10];

        //then create the nodes
        for (int layer = 0; layer < nodes.Length; layer++)
        {
            for (int node = 0; node < nodes[layer].Length; node++)
            {
                nodes[layer][node] = new NNNode();
            }
        }

        //finally, connect weighted edges to the nodes
        for (int layer = 0; layer < nodes.Length - 1; layer++)
        {
            for (int nIn = 0; nIn < nodes[layer].Length; nIn++)
            {
                NNEdge[] edgesIn = new NNEdge[nodes[layer].Length];
                NNEdge[] edgesOut = new NNEdge[nodes[layer + 1].Length];
                for (int nOut = 0; nOut < nodes[layer + 1].Length; nOut++)
                {
                    NNEdge edge = new NNEdge();

                    //connect edge to nodes
                    edge.nodeIn = nodes[layer][nIn];
                    edge.nodeOut = nodes[layer + 1][nOut];

                    //add edge to collection (for init. weights)
                    edges.Add(edge);

                    edge.nodeIn.edgesOut.Add(edge);
                    edge.nodeOut.edgesIn.Add(edge);
                }
            }
        }
        
    }
}
