import React, { useState } from "react";
import { Form, Button, Container, Row, Col, Spinner } from "react-bootstrap";
import { Link, useNavigate} from 'react-router-dom';
import CryptoJS from 'crypto-js';
import axios from 'axios';
import API_BASE_URL from '../../config';

export default function Signup()  {
    const navigate = useNavigate();
    const [validated, set_Validated] = useState(false);
    const [formData, setFormData] = useState({
      email: "",
      pass: "",
    });
    const loadingScreenStyle = {
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        background: 'rgba(255, 255, 255, 0.7)',
        zIndex: 1000,
        flexDirection: 'column',
      };
    const [showLoading, setShowLoading] = useState(false);

    const submitFn = async (event) => {
        event.preventDefault();
        const form = event.currentTarget;
        if (form.checkValidity() === false) {
            event.preventDefault();
            event.stopPropagation();
        }
        set_Validated(true);

        // call the axios functions here for the other functionalities
        try {
            const hashedPWD = CryptoJS.SHA256(formData.pass).toString();
            const resp = await axios.post(API_BASE_URL + 'login', {
                    email: formData.email,
                    password: hashedPWD,
                }
            );

            if(resp.status === 200){
                localStorage.setItem('token',resp.data.access_token)
                setShowLoading(false);
                alert("Login Successful");
                navigate('/api');
                //window.location.replace('/api')
            }else if (resp.status === 205){
                setShowLoading(false);
                alert("Wrong Credentials. Please try again")
                navigate('/login')
                //window.location.replace('/login')
            }
            
        } catch (error) {
            setShowLoading(false);
            alert("Error: ")
            console.error('Error:', error);
        }
    };

    const chngFn = (event) => {
        const { name, value } = event.target;
        setFormData({
            ...formData,
            [name]: value,
        });
    };
    
    return (
      <>
      <div className="about-title">
            <h2>Login</h2>
      </div>
      {showLoading ? (
        <div style={loadingScreenStyle} className="loading-screen">
            <Spinner animation="border" role="status">
            <span className="sr-only">Loading...</span>
            </Spinner>
        </div>
    ) : (
        <Container className="mt-1">
            <Row>
                <Col
                    md={{
                        span: 6,
                        offset: 3,
                    }}
                >
                    <Form noValidate validated={validated} onSubmit={submitFn}>
                      <Form.Group controlId="email">
                            <Form.Label>Email</Form.Label>
                            <Form.Control
                                type="email"
                                name="email"
                                value={formData.email}
                                onChange={chngFn}
                                required
                                isInvalid={
                                    validated &&
                                    !/^\S+@\S+\.\S+$/.test(formData.email)
                                }
                            />
                            <Form.Control.Feedback type="invalid">
                                Please enter a valid email address.
                            </Form.Control.Feedback>
                        </Form.Group>
                        <br/>
                        <Form.Group controlId="password">
                            <Form.Label>Password</Form.Label>
                            <Form.Control
                                type="password"
                                name="pass"
                                value={formData.pass}
                                onChange={chngFn}
                                minLength={6}
                                required
                                isInvalid={
                                    validated && formData.pass.length<0
                                }
                            />
                            <Form.Control.Feedback type="invalid">
                                Please enter your password
                            </Form.Control.Feedback>
                        </Form.Group>
                        <br/>
                        <div class="col-md-12 text-center">
                          <Button variant="outline-primary" type="submit">Submit</Button>
                          <br/><br/>
                          <span className='form-input-login'>Don't have an account? Signup <Link to='/signup'>here</Link></span>
                        </div>
                    </Form>
                </Col>
            </Row>
        </Container>
        )}
        </>
    );
};