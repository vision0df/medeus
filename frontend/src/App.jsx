import { useState } from "react";
import "./App.css";

function App() {
  const [age, setAge] = useState("");
  const [gender, setGender] = useState("");
  const [file, setFile] = useState(null);
  const [recommendation, setRecommendation] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!age || !gender || !file) {
      alert("Пожалуйста, заполните все поля и загрузите файл!");
      return;
    }

    const formData = new FormData();
    formData.append("age", age);
    formData.append("gender", gender);
    formData.append("file", file);

    setLoading(true);
    try {
      const response = await fetch("https://medeus.onrender.com/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Ошибка сервера");

      const data = await response.json();
      setRecommendation(data.text || "Нет рекомендаций");
    } catch (err) {
      console.error(err);
      alert("Ошибка при отправке данных.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="card">
        <h1>Medeus Analyzer</h1>

        <form onSubmit={handleSubmit}>
          <input
            type="number"
            min="1"
            max="112"
            value={age}
            onChange={(e) => setAge(e.target.value)}
            placeholder="Возраст"
          />

          <select value={gender} onChange={(e) => setGender(e.target.value)}>
            <option value="">Выберите пол</option>
            <option value="male">Мужской</option>
            <option value="female">Женский</option>
          </select>

          <input
            type="file"
            accept=".pdf,.jpg,.jpeg,.png"
            onChange={(e) => setFile(e.target.files[0])}
          />

          <button type="submit" disabled={loading}>
            {loading ? "Обработка..." : "Получить рекомендации"}
          </button>
        </form>

        {recommendation && (
          <div className="recommendation">
            <h2>Рекомендации:</h2>
            <p>{recommendation}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;