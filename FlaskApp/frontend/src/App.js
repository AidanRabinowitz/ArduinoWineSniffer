// App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './Home'; // Import Home component
import Predict from './Predict'; // Import Predict component

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
