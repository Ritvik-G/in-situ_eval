import React from 'react';
import { Container, Row, Col } from 'react-bootstrap';
import { FaLinkedinIn, FaGithub } from 'react-icons/fa';

function Footer() {
  return (
    <footer className="bg text-center text-muted ">
      <Container>
        <hr />
        <Row className="justify-content-center">
          <Col xs="auto">
            <span>© AIISC-Devkit - {new Date().getFullYear()}.</span>
          </Col>
        </Row>
      </Container>
    </footer>
  );
}

export default Footer;
