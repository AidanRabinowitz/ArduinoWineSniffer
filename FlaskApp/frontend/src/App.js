import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import axios from 'axios';
import LoadingScreen from './screens/LoadingScreen';

const Home = () => {
    const navigate = useNavigate();
    const [wineName, setWineName] = useState('');
    const [loading, setLoading] = useState(false);

    const handleRunTest = async () => {
        setLoading(true);
        try {
            await axios.post('http://127.0.0.1:5000/run-test', { wine_name: wineName });
            navigate('/predict');  // Navigate to the predict page
        } catch (error) {
            console.error('Error running the test:', error);
        } finally {
            setLoading(false);
        }
    };

       // If loading, return the LoadingScreen
       if (loading) {
        return <LoadingScreen />;
    }

    // Otherwise, return the main UI
    return (
        <div className="min-h-screen flex items-center justify-center bg-gradient-to-r from-purple-400 via-pink-500 to-red-500">
            <div className="bg-white rounded-lg shadow-lg p-8 max-w-lg w-full text-center">
                <h1 className="text-3xl font-bold text-gray-800 mb-4">Smell Wine</h1>
                <input
                    type="text"
                    value={wineName}
                    onChange={(e) => setWineName(e.target.value)}
                    placeholder="Enter wine name"
                    className="border p-2 mb-4 w-full"
                />
                <button
                    onClick={handleRunTest}
                    className="bg-purple-600 text-white font-bold py-2 px-4 rounded"
                >
                    Start Testing
                </button>
            </div>
        </div>
    );
};

const Predict = () => {
    const [predictions, setPredictions] = useState([]);
    const [modalClass, setModalClass] = useState('');
    const [labelAccuracy, setLabelAccuracy] = useState(0);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await axios.get('http://127.0.0.1:5000/predict');
                setPredictions(response.data.predictions);
                setModalClass(response.data.modal_class);
                setLabelAccuracy(response.data.label_accuracy);
            } catch (error) {
                console.error('Error fetching data:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, []);

    if (loading) {
        return <LoadingScreen />; // Show loading screen while fetching predictions
    }

    return (
        <div className="min-h-screen flex items-center justify-center bg-gradient-to-r from-purple-400 via-pink-500 to-red-500">
            <div className="bg-white rounded-lg shadow-lg p-8 max-w-lg w-full">
                <h1 className="text-3xl font-bold text-center text-gray-800 mb-4">Wine Classifications</h1>
                <h2 className="text-xl font-semibold text-gray-700 mb-2">
                    Modal Classification: 
                    <span className="text-purple-600"> {modalClass}</span>
                </h2>
                <h3 className="text-lg font-medium text-gray-600 mb-2">Predicted Classes:</h3>
                <ul className="list-disc list-inside">
                    {predictions.map((className, index) => (
                        <li key={index} className="text-gray-800 mb-1">{className}</li>
                    ))}
                </ul>
                <h4 className="text-lg font-medium text-gray-600 mt-4">
                    Label Accuracy: <span className="text-purple-600">{labelAccuracy.toFixed(2)}%</span>
                </h4>
            </div>
        </div>
    );
};

const App = () => {
    return (
        <Router>
            <Routes>
                <Route path="/predict" element={<Predict />} />
                <Route path="/" element={<Home />} />
            </Routes>
        </Router>
    );
};

export default App;
