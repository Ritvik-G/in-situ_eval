// App.js
import 'bootstrap/dist/css/bootstrap.min.css';
import { BrowserRouter as Router, Routes, Route} from 'react-router-dom'
import Navbar from './Components/Navbar/Navbar';
import Footer from './Components/Footer/Footer';
import Home from './Pages/Home/Home';
import API from './Pages/API/API';

function App() {
  // Assuming you have some state to track authentication status

  return (
    <div className="App">
      <Router>
        <Navbar />
        <Routes>
          <Route element={<Home />} path="/" />
          <Route element={<API />} path="/api" />
        </Routes>
        <Footer />
      </Router>
    </div>
  );
}

export default App;