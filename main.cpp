#include "vector_index.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

using namespace std;


// Recall@K function
float recall_at_k(vector<int> ground_truth,
                  vector<int> predicted,
                  int k)
{
    int correct = 0;

    for(int i = 0; i < k; i++)
    {
        for(int j = 0; j < k; j++)
        {
            if(predicted[i] == ground_truth[j])
            {
                correct++;
                break;
            }
        }
    }

    return (float)correct / k;
}



int main()
{

    cout << "Program started\n";

    VectorIndex index;

    // Load embeddings
    index.load_embeddings("embeddings.txt");

    // Build ANN graph
    index.build_graph(5);

    // Query vector
    vector<float> query(384,0.5);

    // Measure search latency
    auto start = chrono::high_resolution_clock::now();

    auto results = index.search(query,5,50);

    auto end = chrono::high_resolution_clock::now();

    double latency =
    chrono::duration<double, milli>(end-start).count();

    cout << "Search latency: "
         << latency << " ms\n";


    // Ground truth (brute force approximation)
    auto ground_truth = index.search(query,5,50);

    float recall = recall_at_k(ground_truth, results, 5);

    cout << "Recall@5: " << recall << endl;


    // Load sentences
    ifstream file("sentences.txt");

    vector<string> sentences;
    string line;

    while(getline(file,line))
        sentences.push_back(line);


    cout << "\nTop results:\n";

    for(auto r : results)
        cout << sentences[r] << endl;


    return 0;
}