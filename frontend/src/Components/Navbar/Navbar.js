import Container from 'react-bootstrap/Container';
import Nav from 'react-bootstrap/Nav';
import Navbar from 'react-bootstrap/Navbar';
import Offcanvas from 'react-bootstrap/Offcanvas';
import { Link, useNavigate} from "react-router-dom";
import './Navbar.css';
import { useEffect, useState } from 'react';

export default function Navbar_Function(){

    const [localStorageItem, setLocalStorageItem] = useState(false);
    const navigate = useNavigate();

    useEffect(()=>{
        setLocalStorageItem(localStorage.getItem('token') !== null && localStorage.getItem('token').trim() !== '');
    })

    function logout(){
        localStorage.removeItem('token');
        alert("Logged out of the system. Thank you for using our service.")
        navigate('/login');
    }
    return(
        <>
                <Navbar key='md' variant="light">
                <Container fluid className='navbar'>
                    <Navbar.Brand href="/#">
                        <p>AIISC - DevKit</p>
                    </Navbar.Brand>
                    <Navbar.Toggle aria-controls={`offcanvasNavbar-expand-md`} />
                    <Navbar.Offcanvas 
                    id={`offcanvasNavbar-expand-md`} 
                    aria-labelledby={`offcanvasNavbarLabel-expand-md`} 
                    placement="end"
                    >
                    <Offcanvas.Header closeButton>
                        <Offcanvas.Title id={`offcanvasNavbarLabel-expand-md`}>
                        NeuroLit Navigator
                        </Offcanvas.Title>
                    </Offcanvas.Header>
                    <Offcanvas.Body>
                        <Nav className="justify-content-end flex-grow-1 ">
                        <Nav.Link  onClick={()=>navigate('/')} >Home</Nav.Link>

                        <Nav.Link hidden={!localStorageItem} onClick={()=>navigate('/api')} >Query</Nav.Link>
                        <Nav.Link  hidden={!localStorageItem} onClick={()=>navigate('/updateAPI')} >Update API Key</Nav.Link>
                        <Nav.Link hidden={!localStorageItem} onClick={logout} >Logout</Nav.Link>

                        <Nav.Link hidden={localStorageItem} onClick={()=>navigate('/login')}>Login</Nav.Link>
                        <Nav.Link hidden={localStorageItem} onClick={()=>navigate('/signup')}>Signup</Nav.Link>
                        &emsp; &emsp; &emsp;
                        </Nav>
                    </Offcanvas.Body>
                    </Navbar.Offcanvas>
                </Container>
                </Navbar>
        </>
    );
}