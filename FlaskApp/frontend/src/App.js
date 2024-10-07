// app.js
import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import axios from 'axios';
import LoadingScreen from './Components/LoadingScreen'; 

const Home = () => {
    const navigate = useNavigate();
    const [wineName, setWineName] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleRunTest = async () => {
        if (!wineName.trim()) {
            setError("Please enter a wine name.");
            return;
        }

        setLoading(true);
        setError('');
        try {
            const response = await axios.post('http://127.0.0.1:5000/run-test', { wine_name: wineName });
            // Assuming the response contains predictions and accuracy
            if (response.data) {
                navigate('/predict', { state: { predictions: response.data.predictions, label_accuracy: response.data.label_accuracy } });
            }
        } catch (error) {
            console.error('Error running the test:', error);
            setError(error.response?.data?.error || 'An unexpected error occurred.');
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
                {error && <p className="text-red-500 mb-4">{error}</p>}
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
    const navigate = useNavigate();
    const location = useLocation();
    const [predictions, setPredictions] = useState([]);
    const [labelAccuracy, setLabelAccuracy] = useState(0);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    // Access state passed from navigation
    React.useEffect(() => {
        if (location.state && location.state.predictions) {
            const { predictions, label_accuracy } = location.state;
            setPredictions(predictions);
            setLabelAccuracy(label_accuracy);
            setLoading(false);
        }
    }, [location.state]);

    const handleRunAnotherTest = () => {
        navigate('/');
    };

    if (loading) {
        return <LoadingScreen />; // Show loading screen while fetching predictions
    }

    return (
        <div className="min-h-screen flex items-center justify-center bg-gradient-to-r from-purple-400 via-pink-500 to-red-500">
            <div className="bg-white rounded-lg shadow-lg p-8 max-w-lg w-full">
                <h1 className="text-3xl font-bold text-center text-gray-800 mb-4">Wine Classifications</h1>
                <h2 className="text-xl font-semibold text-gray-700 mb-2">
                    Label Accuracy: <span className="text-purple-600">{labelAccuracy.toFixed(2)}%</span>
                </h2>
                <h3 className="text-lg font-medium text-gray-600 mb-2">Predicted Classes:</h3>
                <ul className="list-disc list-inside">
                    {predictions.map((className, index) => (
                        <li key={index} className="text-gray-800 mb-1">{className}</li>
                    ))}
                </ul>
                <button
                    onClick={handleRunAnotherTest}
                    className="mt-6 bg-purple-600 text-white font-bold py-2 px-4 rounded"
                >
                    Run Another Test
                </button>
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
