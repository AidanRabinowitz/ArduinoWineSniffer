import React, { useEffect, useState } from 'react';
import axios from 'axios';

const App = () => {
    const [predictions, setPredictions] = useState([]);
    const [modalClass, setModalClass] = useState('');

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await axios.get('http://127.0.0.1:5000/predict');
                setPredictions(response.data.predictions);
                setModalClass(response.data.modal_class);
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        };

        fetchData();
    }, []);

    return (
        <div style={{ padding: '20px' }}>
            <h1>Wine Classifications</h1>
            <h2>Modal Classification: {modalClass}</h2>
            <h3>Predicted Classes:</h3>
            <ul>
                {predictions.map((className, index) => (
                    <li key={index}>{className}</li>
                ))}
            </ul>
        </div>
    );
};

export default App;
