"use client";

import { useState } from "react";
import axios from "axios";
import "./InputStyles.css"

export default function TransactionForm() {

  const [formData, setFormData] = useState({
    trans_date_trans_time: "",
    cc_num: "",
    merchant: "",
    category: "",
    amt: "",
    gender: "",
    city_pop: "",
    dob: "",
    trans_num: "",
  });

  console.log("Submiited form",formData);

  const [result, setResult] = useState("");

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
  e.preventDefault();

  try {
    const res = await axios.post(
      process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000/predict",
      formData
    );

    const { fraud_probability, prediction } = res.data;

    // Save formatted result as a string
    setResult(
      `Fraud Probability: ${fraud_probability.toFixed(3)}\nClassification: ${prediction}`
    );
  } catch (err) {
    console.error("Error:", err);
    setResult("An error occurred while processing.");
  }
};

  return (
  <form onSubmit={handleSubmit} className="form-container">
  <h1 className="form-title">Transaction Input Form</h1>

  <InputField
    label="Transaction Date & Time"
    name="trans_date_trans_time"
    type="datetime-local"
    required
    onChange={handleChange}
  />

  <InputField
    label="Credit Card Number"
    name="cc_num"
    type="number"
    required
    pattern="\d{16}"
    placeholder="16-digit card number"
    onChange={handleChange}
  />

  <InputField
    label="Merchant"
    name="merchant"
    type="text"
    required
    onChange={handleChange}
  />

  <InputField
    label="Category"
    name="category"
    type="text"
    required
    placeholder="e.g., personal_care"
    onChange={handleChange}
  />

  <InputField
    label="Amount"
    name="amt"
    type="number"
    step="0.01"
    required
    onChange={handleChange}
  />

  <div className="form-group">
    <label htmlFor="gender" className="form-label">Gender</label>
    <select
      title="gender"
      name="gender"
      onChange={handleChange}
      required
      className="form-select"
    >
      <option value="">Select Gender</option>
      <option value="M">Male</option>
      <option value="F">Female</option>
    </select>
  </div>

  <InputField
    label="City Population"
    name="city_pop"
    type="number"
    required
    onChange={handleChange}
  />

  <InputField
    label="Date of Birth"
    name="dob"
    type="date"
    required
    onChange={handleChange}
  />

  <InputField
    label="Transaction Number"
    name="trans_num"
    type="text"
    required
    placeholder="e.g., 2da90c7d74bd46a0caf3777415b3ebd3"
    onChange={handleChange}
  />

  <button type="submit" className="submit-button">Submit</button>

  {result && (
    <div className="result-section">
      <h2 className="result-title">Prediction Result:</h2>
      <p className="result-text">{result}</p>
    </div>
  )}
</form>

  );
}

type InputFieldProps = {
  label: string;
  name: string;
  type: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  placeholder?: string;
  required?: boolean;
  pattern?: string;
  step?: string;
};

function InputField({
  label,
  name,
  type,
  onChange,
  placeholder,
  required,
  pattern,
  step,
}: InputFieldProps) {
  return (
    <div className="flex flex-col">
      <label htmlFor={name} className="mb-1">
        {label}
      </label>
      <input
        name={name}
        type={type}
        required={required}
        onChange={onChange}
        placeholder={placeholder}
        pattern={pattern}
        step={step}
        className="border px-3 py-2 rounded-md"
      />
    </div>
  );
}
