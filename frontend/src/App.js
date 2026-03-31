import React, { useState } from "react";
import UploadForm from "./components/UploadForm";

function App() {
  const [analysis, setAnalysis] = useState("");
  const [loading, setLoading] = useState(false);

  const handleOcrSubmit = async (formData, { age, gender }) => {
    setLoading(true);
    try {
      // 1️⃣ Отправляем файл на OCR
      const ocrResponse = await fetch("http://localhost:5000/ocr", {
        method: "POST",
        body: formData,
      });
      const ocrResult = await ocrResponse.json();

      // 2️⃣ Отправляем текст сразу на LLM (Gemini)
      const analyzeResponse = await fetch("http://localhost:5000/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: ocrResult.text, age, gender }),
      });
      const analyzeResult = await analyzeResponse.json();

      // 3️⃣ Показываем пользователю только расшифровку
      setAnalysis(analyzeResult.analysis);
    } catch (error) {
      console.error(error);
      alert("Ошибка при обработке файла");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>Medeus — Расшифровка анализов</h1>
      <UploadForm onOcrSubmit={handleOcrSubmit} />
      <hr />
      {loading ? (
        <p>Обработка файла… Пожалуйста, подождите</p>
      ) : (
        <>
          <h2>Расшифровка и рекомендации:</h2>
          <pre>{analysis}</pre>
        </>
      )}
    </div>
  );
}

export default App;