// Screens/loadingscreen.js
import React, { useEffect, useState } from 'react';

const LoadingScreen = () => {
    const loadingImage = process.env.PUBLIC_URL + "/images/blacktieimage.png"; // Ensure correct path
    const [loadingText, setLoadingText] = useState("Loading");

    useEffect(() => {
        const interval = setInterval(() => {
            setLoadingText(prev => {
                // Cycle through the dots
                if (prev.length < 3) {
                    return prev + ".";
                }
                return "Loading"; // Reset back to "Loading"
            });
        }, 500); // Adjust the time as needed

        return () => clearInterval(interval);
    }, []);

    return (
        <div className="min-h-screen flex flex-col items-center justify-center bg-white">
            <img
                src={loadingImage}
                alt="Loading"
                className="animate-spin w-48 h-48 mb-4" // Added margin for spacing
            />
            <div className="text-xl text-gray-700">{loadingText}</div> {/* Added loading text */}
        </div>
    );
};

export default LoadingScreen;
