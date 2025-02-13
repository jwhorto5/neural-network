using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;

public class AppController : MonoBehaviour
{
    private List<int[]> testingDataList = new List<int[]>();
    private List<int[]> trainingDataList = new List<int[]>();

    private int[] currentInputs;
    private float[] outputsCorrectness;
    
    public int layers; //number of layers, including input and output layer
    public int hiddenLayerLength; //the length of each hidden layer

    public int dataIndex;
    public float learningRate = 0.1f;
    public int generations = 10;

    private int numberGuess;
    private int numCorrectGuesses = 0;

    public GameObject imageGrid;
    public Image[] images;
    public Text resultsText;

    void Start()
    {
        outputsCorrectness = new float[10];

        images = imageGrid.GetComponentsInChildren<Image>();

        //read the file
        ReadFile("/optdigits_train.txt", trainingDataList);
        ReadFile("/optdigits_test.txt", testingDataList);

        //create the neural network
        NeuralNetwork neuralNetwork = new NeuralNetwork(layers, hiddenLayerLength);

        //initialize weights
        InitWeights(neuralNetwork);

        //make an array to store the test accuracies
        float[] accuracies = new float[generations];


        for (int gens = 0; gens < generations; gens++)
        {
            numCorrectGuesses = 0;
            for (int i = 0; i < trainingDataList.Count; i++)
            {
                //show the data visually
                //PrintData(i);
                //DisplayData(i);
                SetCorrectAnswer(i);

                FeedForward(neuralNetwork, trainingDataList, i, true);
            }

            Debug.Log("Accuracy: " + ((float)numCorrectGuesses / (float)trainingDataList.Count));

            numCorrectGuesses = 0;
            for (int i = 0; i < testingDataList.Count; i++)
            {
                //show the data visually
                //PrintData(i);
                //DisplayData(i);
                SetCorrectAnswer(i);

                FeedForward(neuralNetwork, testingDataList, i, false);
            }

            accuracies[gens] = (float)numCorrectGuesses / (float)testingDataList.Count;
            Debug.Log("Accuracy: " + accuracies[gens]);
        }

        CreateGraphData(accuracies);

        resultsText.text =
            "Training complete after " + generations + " generations"
            + "\nHighest accuracy: " + Mathf.Max(accuracies);
    }

    void ReadFile(string path, List<int[]> outputList)
    {
        StreamReader sr = new StreamReader(Application.persistentDataPath + path);

        try
        {
            while (!sr.EndOfStream)
            {
                string str = sr.ReadLine();

                string[] nums = str.Split(',');
                int[] dataChunk = new int[nums.Length];
                for (int i = 0; i < nums.Length; i++)
                {
                    dataChunk[i] = int.Parse(nums[i]);
                }
                outputList.Add(dataChunk);
            }

            sr.Close();
        }
        catch (IOException ioe)
        {
            Debug.Log("Error with I/O:\n" + ioe.Message);
        }
    }

    void CreateGraphData(float[] arr)
    {
        StreamWriter sw = new StreamWriter(Application.persistentDataPath + "/graph_data.csv");

        try
        {
            foreach (float value in arr)
            {
                sw.Write(value + "\n");
            }
        }
        catch (IOException ioe)
        {
            Debug.Log("Error with I/O:\n" + ioe.Message);
        }

        sw.Close();
    }

    void InitWeights(NeuralNetwork nn)
    {
        //initialize weights
        foreach (NNEdge edge in nn.edges)
            edge.weight = Random.Range(0.001f, 0.01f);
    }

    void FeedForward(NeuralNetwork nn, List<int[]> dataList, int dataIndex, bool backProp)
    {
        //assign data values to input layer
        for (int n = 0; n < nn.nodes[0].Length; n++)
        {
            nn.nodes[0][n].value = dataList[dataIndex][n];
        }

        //feed forward
        for (int l = 1; l < nn.nodes.Length; l++)
        {
            for (int n = 0; n < nn.nodes[l].Length; n++)
            {
                NNNode node = nn.nodes[l][n];

                node.sum = 0f;
                foreach (NNEdge edge in node.edgesIn)
                {
                    node.sum += edge.weight * edge.nodeIn.value;
                }
                node.value = Sigmoid(node.sum);
            }
        }

        //printing values (for debugging)
        float bestMatch = 0;
        for (int i = 0; i < nn.nodes[nn.nodes.Length - 1].Length; i++)
        {
            NNNode node = nn.nodes[nn.nodes.Length - 1][i];
            if (node.value > bestMatch)
            {
                bestMatch = node.value;
                numberGuess = i;
            }
        }

        //string temp = "";
        //for(int i = 0; i < nn.nodes[0].Length; i++)
        //{
        //    for (int j = 0; j < nn.nodes[0][i].weights.Count; j++)
        //    {
        //        temp += nn.nodes[0][i].weights[j] + ", ";
        //    }
        //    temp += "\n";
        //}
        //Debug.Log(temp);
        //PrintOutputs(nn.nodes[nn.layers - 1]);

        if (numberGuess == dataList[dataIndex][64]) numCorrectGuesses++;

        //Debug.Log("this might be " + numberGuess);
        //Debug.Log("Actual answer is " + dataList[dataIndex][64]);
        numberGuess = 0;

        //back propagation (the hard part)

        if (!backProp) return;

        //output layer first
        for (int i = 0; i < nn.nodes[nn.nodes.Length - 1].Length; i++)
        {
            NNNode node = nn.nodes[nn.nodes.Length - 1][i];

            node.error =
                InvSigmoid(node.sum) //inverse of sigmoid function
                * (outputsCorrectness[i] - node.value);
        }

        //then the rest
        for (int i = nn.nodes.Length - 2; i >= 0; i--)
        {
            for (int j = 0; j < nn.nodes[i].Length; j++)
            {
                NNNode currentNode = nn.nodes[i][j];

                float sum = 0f;
                foreach (NNEdge edge in currentNode.edgesOut)
                {
                    sum += edge.weight * edge.nodeOut.error;
                }
                currentNode.error =
                    InvSigmoid(currentNode.sum)
                    * sum;
            }
        }

        foreach (NNEdge edge in nn.edges)
        {
            edge.weight += learningRate * edge.nodeIn.value * edge.nodeOut.error;
        }
    }

    void SetCorrectAnswer(int index)
    {
        for (int o = 0; o < outputsCorrectness.Length; o++)
        {
            outputsCorrectness[o] = 0f;
        }

        outputsCorrectness[trainingDataList[index][64]] = 1f;
    }

    float Sigmoid(float x)
    {
        return 1 / (1 + Mathf.Exp(-x));
    }

    float InvSigmoid(float x)
    {
        return Sigmoid(x) * Sigmoid(1 - x);
    }

    void PrintData(int index)
    {
        string str = "";
        foreach (int data in trainingDataList[index])
        {
            str += data + ", ";
        }

        Debug.Log(str);
    }

    void PrintOutputs(NNNode[] outputs)
    {
        string str = "";
        foreach (NNNode output in outputs)
        {
            str += output.value + ", ";
        }

        Debug.Log(str);
    }

    void DisplayData(int index)
    {
        for (int i = 0; i < images.Length; i++)
        {
            float value = (float) trainingDataList[index][i] / 16;
            images[i].color = new Color(value, value, value);
        }
    }
}
