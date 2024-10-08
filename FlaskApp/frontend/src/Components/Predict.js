import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import LoadingScreen from './screens/LoadingScreen'; // Ensure you have this component

const Predict = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const [predictions, setPredictions] = useState([]);
    const [labelAccuracy, setLabelAccuracy] = useState(0);
    const [winePercentages, setWinePercentages] = useState({});
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    // Access state passed from navigation
    useEffect(() => {
        if (location.state && location.state.predictions) {
            const { predictions, label_accuracy } = location.state;
            setPredictions(predictions);
            setLabelAccuracy(label_accuracy);
            calculateWinePercentages(predictions);
            setLoading(false);
        }
    }, [location.state]);

    // Function to calculate percentages of each wine in the predictions array
    const calculateWinePercentages = (predictions) => {
        const wineCounts = {};
        const totalPredictions = predictions.length;

        predictions.forEach((wine) => {
            if (wineCounts[wine]) {
                wineCounts[wine] += 1;
            } else {
                wineCounts[wine] = 1;
            }
        });

        // Convert counts to percentages
        const percentages = {};
        Object.keys(wineCounts).forEach((wine) => {
            percentages[wine] = ((wineCounts[wine] / totalPredictions) * 100).toFixed(2);
        });

        setWinePercentages(percentages);
    };

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
                <h3 className="text-lg font-medium text-gray-600 mb-4">Predicted Wine Class Percentages:</h3>

                {/* Table to display wine percentages */}
                <table className="min-w-full bg-white border border-gray-300 mb-4">
                    <thead>
                        <tr>
                            <th className="py-2 px-4 border-b">Wine Name</th>
                            <th className="py-2 px-4 border-b">% of Predictions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {Object.keys(winePercentages).map((wineName, index) => (
                            <tr key={index}>
                                <td className="py-2 px-4 border-b">{wineName}</td>
                                <td className="py-2 px-4 border-b">{winePercentages[wineName]}%</td>
                            </tr>
                        ))}
                    </tbody>
                </table>

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
