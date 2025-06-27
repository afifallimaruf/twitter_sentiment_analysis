import { useState } from "react";
import "./App.css";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await fetch("http://0.0.0.0:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await response.json();
      setResult(data);
      console.log(result);
    } catch (error) {
      setResult({ error });
    }
    setLoading(false);
  };

  return (
    <>
      <div className="min-h-screen bg-gray-100 flex-center justify-center">
        <div className="bg-white p-8 rounded-lg shadow-md w-full max-w-md">
          <h1 className="text-md text-black font-bold mb-6 text-center">
            Sentiment Analysis
          </h1>
          <form onSubmit={handleSubmit} className="space-y-4">
            <textarea
              className="w-full p-3 text-black border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Enter text to analyze"
              rows="4"
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
            <button
              type="submit"
              className="w-full bg-blue-500 text-white p-3 rounded-md hover:bg-blue-600 disabled:bg-blue-300"
              disabled={loading || !text}
              onClick={handleSubmit}
            >
              {loading ? "Analyzing..." : "Analyze Sentiment"}
            </button>
          </form>
          {result && (
            <div className="mt-6 p-4 bg-gray-50 rounded-md text-black">
              {result.error ? (
                <p className="text-red-500">{result.error}</p>
              ) : (
                <>
                  <p>
                    <strong>Text:</strong>
                    {result.text}
                  </p>
                  <p>
                    <strong>Sentiment:</strong>
                    {result.sentiment}
                  </p>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  );
}

export default App;
