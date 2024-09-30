import React, { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [classifiedWines, setClassifiedWines] = useState([]);
  const [modalClass, setModalClass] = useState('');

  // Automatically fetch classification data when the component mounts
  useEffect(() => {
    axios.post('http://127.0.0.1:5000/classify', {}) // Empty post request to trigger Flask endpoint
      .then(response => {
        setClassifiedWines(response.data.classified_wines);
        setModalClass(response.data.modal_class);
      })
      .catch(error => {
        console.error('Error fetching the classified wines:', error);
      });
  }, []); // Empty dependency array to only run once on component mount

  return (
    <div className="App">
      <h1>Wine Classification Results</h1>

      {/* Display the list of classified wines */}
      {classifiedWines.length > 0 && (
        <div>
          <h2>Classified Wines:</h2>
          <ul>
            {classifiedWines.map((wine, index) => (
              <li key={index}>{wine}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Display modal classification */}
      {modalClass && (
        <div>
          <h2>Modal Classification: {modalClass}</h2>
        </div>
      )}
    </div>
  );
}

export default App;
