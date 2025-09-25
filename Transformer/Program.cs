using TransformersSharp;

var transformer = SentenceTransformer.FromModel("sentence-transformers/clip-ViT-B-32-multilingual-v1", trustRemoteCode: true);
var sentences = new List<string>
{
    "The quick brown fox jumps over the lazy dog.",
    "Transformers are amazing for natural language processing."
};
var embeddings = await transformer.GenerateAsync(sentences);

foreach (var embedding in embeddings)
{
    Console.WriteLine($"Vector: {string.Join(", ", embedding.Vector.ToArray().Take(10))}...");
}