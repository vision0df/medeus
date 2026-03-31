import React, { useState } from "react";

const BACKEND_URL = "https://medeus.onrender.com";

export default function AnalyzeForm() {
  const [file, setFile] = useState(null);
  const [age, setAge] = useState("");
  const [gender, setGender] = useState("мужской");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file || !age || !gender) {
      alert("Заполните все поля и выберите файл");
      return;
    }

    setLoading(true);
    setResult("");

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("age", age);
      formData.append("gender", gender);

      const response = await fetch(BACKEND_URL, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (data.error) {
        alert("Ошибка: " + data.error);
      } else {
        setResult(data.analysis);
      }
    } catch (err) {
      console.error(err);
      alert("Произошла ошибка при отправке файла");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: "500px", margin: "auto" }}>
      <h2>Medeus — расшифровка анализов</h2>
      <form onSubmit={handleSubmit}>
        <div>
          <label>Возраст:</label>
          <input
            type="number"
            value={age}
            onChange={(e) => setAge(e.target.value)}
            required
          />
        </div>

        <div>
          <label>Пол:</label>
          <select value={gender} onChange={(e) => setGender(e.target.value)}>
            <option value="мужской">мужской</option>
            <option value="женский">женский</option>
          </select>
        </div>

        <div>
          <label>Файл (PDF/изображение до 1 МБ):</label>
          <input
            type="file"
            accept=".pdf,image/*"
            onChange={(e) => setFile(e.target.files[0])}
            required
          />
        </div>

        <button type="submit" disabled={loading}>
          {loading ? "Обрабатываем..." : "Отправить"}
        </button>
      </form>

      {result && (
        <div style={{ marginTop: "20px" }}>
          <h3>Расшифровка и рекомендации:</h3>
          <p>{result}</p>
        </div>
      )}
    </div>
  );
}