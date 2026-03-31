import React, { useState } from "react";

function UploadForm({ onOcrSubmit }) {
  const [file, setFile] = useState(null);
  const [age, setAge] = useState("");
  const [gender, setGender] = useState("male");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return alert("Загрузите файл");

    const formData = new FormData();
    formData.append("file", file);

    // Передаём данные родителю
    onOcrSubmit(formData, { age, gender });
  };

  return (
    <form onSubmit={handleSubmit}>
      <label>
        Пол:
        <select value={gender} onChange={(e) => setGender(e.target.value)}>
          <option value="male">Мужской</option>
          <option value="female">Женский</option>
        </select>
      </label>
      <br />
      <label>
        Возраст:
        <input
          type="number"
          value={age}
          onChange={(e) => setAge(e.target.value)}
        />
      </label>
      <br />
      <label>
        Файл (Фото или PDF):
        <input type="file" onChange={(e) => setFile(e.target.files[0])} />
      </label>
      <br />
      <button type="submit">Отправить на OCR</button>
    </form>
  );
}

export default UploadForm;