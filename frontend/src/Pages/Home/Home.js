import React, {useState, useEffect} from 'react';
import Button from 'react-bootstrap/Button';
import Offcanvas from 'react-bootstrap/Offcanvas';
import './Home.css';

export default function Home() {
    const [openCanvas, setOpenCanvas] = useState(null);
    const handleToggle = (id) => {
        setOpenCanvas((prev) => (prev === id ? null : id));
    };
    return (
        <>
        <div className="home-container">
            <div className='left-container'>
                <Button variant="outline-primary" onClick={() => handleToggle("config")} className="me-2">Groq Config</Button>
                <Offcanvas show={openCanvas === "config"} onHide={() => setOpenCanvas(null)} scroll={true} backdrop={true} placement={'start'}>
                    <Offcanvas.Header closeButton>
                        <Offcanvas.Title>Groq Config</Offcanvas.Title>
                    </Offcanvas.Header>
                    <Offcanvas.Body>
                        Model Config
                    </Offcanvas.Body>
                </Offcanvas>
            </div>
            <div className='right-container'>
                <Button variant="outline-primary" onClick={() => handleToggle("evals")} className="me-2">Eval Stats</Button>
                <Offcanvas show={openCanvas === "evals"} onHide={() => setOpenCanvas(null)} scroll={true} backdrop={true} placement={'end'}>
                    <Offcanvas.Header closeButton>
                        <Offcanvas.Title>Eval Stats</Offcanvas.Title>
                    </Offcanvas.Header>
                    <Offcanvas.Body>
                        Popular Stats
                    </Offcanvas.Body>
                </Offcanvas>
            </div>
        </div>
        <br/>
        <div className='central-container'>
        {/* Central content goes here */}
        <p>hi</p>
        </div>
        </>
    );
}