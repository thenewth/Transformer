using Microsoft.Extensions.AI;
using TransformersSharp;

var transformer_text = SentenceTransformer.FromModel("sentence-transformers/clip-ViT-B-32-multilingual-v1", trustRemoteCode: true);
var transformer_iamge = SentenceTransformer.FromModel("clip-ViT-B-32", trustRemoteCode: true);

var text_embeddings = transformer_text.GenerateSentence("A dog in the snow");
var image_embeddings = transformer_iamge.GenerateImage("https://images.unsplash.com/photo-1547494912-c69d3ad40e7f?ixlib=rb-4.1.0&q=85&fm=jpg&crop=entropy&cs=srgb&w=640");

Console.WriteLine($"Vector: {string.Join(", ", text_embeddings.ToArray().Take(10))}...");
Console.WriteLine($"Vector: {string.Join(", ", image_embeddings.ToArray().Take(10))}...");