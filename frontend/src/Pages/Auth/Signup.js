import React, { useState } from "react";
import { Form, Button, Container, Row, Col, Spinner } from "react-bootstrap";
import { Link, useNavigate} from "react-router-dom";
import CryptoJS from 'crypto-js';
import axios from 'axios';
import API_BASE_URL from '../../config';
export default function Signup()  {
    
    const [validated, set_Validated] = useState(false);
    const navigate = useNavigate();
    const [formData, setFormData] = useState({
      email: "",
      pass: "",
      confimPass: "",
      API: "",
      confirmAPI: ""
    });
    localStorage.removeItem('token');
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
            event.stopPropagation();
        } else {
            setShowLoading(true);
            const hashedPWD = CryptoJS.SHA256(formData.pass).toString();
            try {
                const resp = await axios.post(API_BASE_URL+ 'signup', {
                        email: formData.email,
                        password: hashedPWD,
                        api: formData.API,
                    }
                );

                if(resp.status === 200){
                    localStorage.setItem('token',resp.data.access_token)
                    setShowLoading(false);
                    alert("Signup successful");
                    navigate('/api');
                    //window.location.replace('/api');
                }else if (resp.status === 205){
                    setShowLoading(false);
                    alert("Email already registered");
                    alert("Please login with your registered credentials");
                    navigate('/login');
                    //window.location.replace('/login')
                }
                
            } catch (error) { // Updated this part for the time being
                // setShowLoading(false);
                // alert("An error occured during signup. Please try again.")
                //window.location.replace('/signup')
                navigate('/signup');
                console.error('Error:', error);
            }
        }
        set_Validated(true);
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
            <h2>Signup</h2>
      </div>
      {showLoading ? (
        <div style={loadingScreenStyle} className="loading-screen">
            <Spinner animation="border" role="status">
            <span className="sr-only">Loading...</span>
            </Spinner>
        </div>
    ) : (
      <Container className="mt-1">
            <Row className="roomfac">
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
                                    validated && formData.pass.length < 6
                                }
                            />
                            <Form.Control.Feedback type="invalid">
                                Password must be at least 6 characters long.
                            </Form.Control.Feedback>
                        </Form.Group>
                        <br/>
                        <Form.Group controlId="confirmPassword">
                            <Form.Label>Confirm Password</Form.Label>
                            <Form.Control
                                type="password"
                                name="confimPass"
                                value={formData.confimPass}
                                onChange={chngFn}
                                minLength={6}
                                required
                                pattern={formData.pass}
                                isInvalid={
                                    validated &&
                                    formData.confimPass !== formData.pass
                                }
                            />
                            <Form.Control.Feedback type="invalid">
                                Passwords do not match.
                            </Form.Control.Feedback>
                        </Form.Group>
                        <br/>
                        <Form.Group controlId="API">
                            <Form.Label className="text-center">API</Form.Label>
                            <Form.Control
                                type="text"
                                name="API"
                                value={formData.API}
                                onChange={chngFn}
                                required
                                isInvalid={
                                    validated && formData.API.length < 0
                                }
                            />
                            <Form.Control.Feedback type="invalid">
                                Please enter the API
                            </Form.Control.Feedback>
                        </Form.Group>
                        <br/>
                        <Form.Group controlId="confirmAPI">
                            <Form.Label>Confirm API</Form.Label>
                            <Form.Control
                                type="text"
                                name="confirmAPI"
                                value={formData.confirmAPI}
                                onChange={chngFn}
                                pattern={formData.API}
                                required
                                isInvalid={
                                    validated &&
                                    formData.confirmAPI !== formData.API
                                }
                            />
                            <Form.Control.Feedback type="invalid">
                                APIs do not match.
                            </Form.Control.Feedback>
                        </Form.Group>
                        <br/>
                        <div className="col-md-12 text-center">
                          <Button  variant="outline-primary" type="submit">Submit</Button>
                          <br/><br/>
                          <span className='form-input-login'>Already have an account? Login <Link to='/login'>here</Link></span>
                        </div>
                        </Form>
                </Col>
            </Row>
        </Container>
    )}
        </>
    );
};