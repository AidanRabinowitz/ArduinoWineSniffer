import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import LoadingScreen from './Components/LoadingScreen';

const Predict = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const [wineCounts, setWineCounts] = useState({});
    const [labelAccuracy, setLabelAccuracy] = useState(0);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (location.state && location.state.predictions) {
            const { predictions, label_accuracy } = location.state;

            // Use reduce to count occurrences of each wine label
            const counts = predictions.reduce((acc, wine) => {
                acc[wine] = (acc[wine] || 0) + 1;
                return acc;
            }, {});

            setWineCounts(counts);
            setLabelAccuracy(label_accuracy);
            setLoading(false);
        }
    }, [location.state]);

    const handleRunAnotherTest = () => {
        navigate('/');
    };

    if (loading) {
        return <LoadingScreen />;
    }

    return (
        <div className="min-h-screen flex items-center justify-center bg-gradient-to-r from-purple-400 via-pink-500 to-red-500">
            <div className="bg-white rounded-lg shadow-lg p-8 max-w-lg w-full">
                <h1 className="text-3xl font-bold text-center text-gray-800 mb-4">Wine Classifications</h1>
                <h2 className="text-xl font-semibold text-gray-700 mb-2">
                    Label Accuracy: <span className="text-purple-600">{labelAccuracy.toFixed(2)}%</span>
                </h2>
                <h3 className="text-lg font-medium text-gray-600 mb-2">Predicted Classes:</h3>
                
                {/* Display each wine label with its count */}
                <ul className="list-disc list-inside">
                    {Object.keys(wineCounts).map((wine, index) => (
                        <li key={index} className="text-gray-800 mb-1">
                            {wine} - {wineCounts[wine]} {wineCounts[wine] > 1 ? 'times' : 'time'}
                        </li>
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

export default Predict;
 