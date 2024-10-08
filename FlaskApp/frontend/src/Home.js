// Home.js
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
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

    if (loading) {
        return <LoadingScreen />;
    }

    return (
        <div className="min-h-screen flex items-center justify-center bg-gradient-to-r from-purple-400 via-pink-500 to-red-500 relative">
            {/* Embed GIF with clickable custom URL overlay */}
            <div className="absolute top-0 left-0 m-4 w-[150px] h-[150px]">
                <iframe
                    src="https://giphy.com/embed/1hBYNPkf4d3e5cnl1Y"
                    width="150"
                    height="150"
                    style={{ border: 'none' }}
                    frameBorder="0"
                    className="giphy-embed"
                    allowFullScreen
                ></iframe>

                <a
                    href="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="absolute top-0 left-0 w-full h-full"
                    style={{ zIndex: 10 }}
                ></a>
            </div>

            <div className="bg-white rounded-lg shadow-lg p-8 max-w-lg w-full text-center">
                <h1 className="text-3xl font-bold text-gray-800 mb-4">Smart Wine Sniffer</h1>
                <p className="text-gray-600 mb-6">
                    This innovative robot has been trained on a dataset of wine scents collected over different days, enabling it to recognize and label various wines purely from their scent.
                </p>

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

export default Home;
