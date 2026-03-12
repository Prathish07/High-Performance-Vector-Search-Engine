#include "vector_index.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <queue>
#include <iostream>
#include <algorithm>


// Load embeddings from file
void VectorIndex::load_embeddings(const std::string& filename)
{

    std::ifstream file(filename);

    if(!file.is_open())
    {
        std::cout << "Error: Could not open embeddings file\n";
        return;
    }

    std::string line;
    int id = 0;

    while(getline(file,line))
    {

        std::stringstream ss(line);
        float value;

        std::vector<float> embedding;

        while(ss >> value)
            embedding.push_back(value);

        nodes.push_back({id++, embedding});
    }

    std::cout << "Loaded vectors: "
              << nodes.size() << std::endl;
}



float VectorIndex::cosine_similarity(
    const std::vector<float>& a,
    const std::vector<float>& b)
{

    float dot = 0;
    float normA = 0;
    float normB = 0;

    for(int i = 0; i < a.size(); i++)
    {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    if(normA == 0 || normB == 0)
        return 0;

    return dot / (sqrt(normA) * sqrt(normB));
}



// Build ANN graph
void VectorIndex::build_graph(int M)
{

    graph.resize(nodes.size());

    for(int i = 0; i < nodes.size(); i++)
    {

        std::vector<std::pair<float,int>> sims;

        for(int j = 0; j < nodes.size(); j++)
        {

            if(i == j)
                continue;

            float sim =
            cosine_similarity(nodes[i].embedding,
                              nodes[j].embedding);

            sims.push_back({sim,j});
        }

        std::sort(sims.begin(), sims.end(),
        [](auto &a, auto &b)
        {
            return a.first > b.first;
        });

        for(int k = 0; k < M && k < sims.size(); k++)
            graph[i].push_back(sims[k].second);
    }

    std::cout << "Graph built with "
              << M
              << " neighbors per node\n";
}



// ANN search using efSearch
std::vector<int> VectorIndex::search(
    const std::vector<float>& query,
    int k,
    int efSearch)
{

    std::priority_queue<std::pair<float,int>> candidates;
    std::priority_queue<std::pair<float,int>> topk;

    std::vector<bool> visited(nodes.size(), false);

    int entry = 0;

    float sim =
    cosine_similarity(query, nodes[entry].embedding);

    candidates.push({sim, entry});
    visited[entry] = true;

    while(!candidates.empty())
    {

        auto current = candidates.top();
        candidates.pop();

        int node_id = current.second;

        topk.push(current);

        if(topk.size() > efSearch)
            topk.pop();


        for(int neighbor : graph[node_id])
        {

            if(visited[neighbor])
                continue;

            visited[neighbor] = true;

            float neighbor_sim =
            cosine_similarity(query,
                              nodes[neighbor].embedding);

            candidates.push({neighbor_sim, neighbor});
        }
    }


    std::vector<int> results;

    while(!topk.empty() && results.size() < k)
    {
        results.push_back(topk.top().second);
        topk.pop();
    }

    return results;
}