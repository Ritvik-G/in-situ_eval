import React, { useState } from 'react';
import { Button, Offcanvas, Container, Row, Col, ListGroup, Card, Form, Dropdown, InputGroup } from 'react-bootstrap';
import Badge from 'react-bootstrap/Badge';
import { FaFontAwesome, FaInfo } from 'react-icons/fa';

import AccessTypeSelector from '../../Components/API/AccessSelector';
import ModelSelector from '../../Components/API/ModelSelector';
import RAGSelector from '../../Components/API/RAGSelector';

const Playground = () => {
  const [showOffcanvas, setShowOffcanvas] = useState(false);
  const [model, setModel] = useState('');
  const [accessType, setAccessType] = useState('GROQ');
  const [apikey, setAPIKey] = useState('');
  const [stopSequence, setStopSequence] = useState('');
  const [Stream, setStream] = useState(false);

  const [temperature, setTemperature] = useState(0);

  const [topP, setTopP] = useState(0);

  const [ragType, setRagType] = useState('Vanilla RAG');
  const [chunkSize, setChunkSize] = useState(1000);
  const [chunkOverlap, setChunkOverlap] = useState(100);
  const [topK, setTopK] = useState(3);

  const tokens_min = 0
  const tokens_max = 10000
  const [tokens, setTokens] = useState((tokens_min+tokens_max)/2);


  const handleTempSliderChange = (e) => {
    setTemperature(e.target.value);
  };
  const handleTemperatureInputChange = (e) => {
    let val = e.target.value;
    if (val === "") {
      setTemperature("");
      return;
    }
    val = Number(val);
    if (!isNaN(val) && val >= 0 && val <= 1) {
      setTemperature(val);
    }
  };


  const handleTokenSliderChange = (e) => {
    setTokens(e.target.value);
  };
  const handleTokenInputChange = (e) => {
    let val = e.target.value;
    if (val === "") {
      setTokens("");
      return;
    }
    val = Number(val);
    if (!isNaN(val) && val >= 0 && val <= 1) {
      setTokens(val);
    }
  };


  const handleTopPSliderChange = (e) => {
    setTopP(e.target.value);
  };
  const handleTopPInputChange = (e) => {
    let val = e.target.value;
    if (val === "") {
      setTopP("");
      return;
    }
    val = Number(val);
    if (!isNaN(val) && val >= 0 && val <= 1) {
      setTopP(val);
    }
  };

    const [fileUploadErrorMessage, setFileUploadErrorMessage] = useState('');

    const handleFileChange = (event) => {
        const files = event.target.files;
        let totalSize = 0;
        let isValid = true;

        // Check each file
        for (let i = 0; i < files.length; i++) {
            const file = files[i];

            // Check file type
            if (file.type !== 'application/json') {
                setFileUploadErrorMessage('Only JSON files are allowed.');
                isValid = false;
                break;
            }

            // Accumulate total size
            totalSize += file.size;
        }

        // Check total size
        if (totalSize > 20 * 1024 * 1024) { // 20MB in bytes
            setFileUploadErrorMessage('Total file size exceeds 20MB.');
            isValid = false;
        }

        // If everything is valid, clear any previous error message
        if (isValid) {
            setFileUploadErrorMessage('');
            // Proceed with file upload or further processing
        } else {
            // Clear the file input if validation fails
            event.target.value = '';
        }
    };

    const loadConfiguration = async () => {
      const configData = {
        accessType:accessType,
        model:model,
        apikey : apikey,
        temperature : temperature,
        max_tokens : tokens,
        top_p : topP,
        stop_sequence : stopSequence,
        stream : Stream,
        ragType:ragType,
        chunkSize:chunkSize,
        chunkOverlap:chunkOverlap,
        topK:topK
      }
    }
  
  const sidebarContent = (
    <>
    
    <ListGroup as="ol"  variant='flush'>
        <ListGroup.Item as="li"> 
            <div>
            <div className="fw-bold"><h6><b>Model Configuration &emsp; <Badge bg="secondary" pill> <FaInfo/> </Badge></b></h6></div>
            <Form>
                <Form.Group>
                    <Form.Label>API Key</Form.Label>
                    <Form.Control type='password' value={apikey} onChange={(e) => setAPIKey(e.target.value)} placeholder='Enter your API key'></Form.Control>
                </Form.Group>
                <br/>
                <Form.Group>
                    <InputGroup className="mb-3 d-flex justify-content-between align-items-center">
                    <Form.Label>Temperature &emsp;</Form.Label>
                    <Form.Control type="number" value={temperature} onChange={handleTemperatureInputChange} min={0} max={1} size='sm' />
                    <Form.Range min={0} max={1} step={0.01} value={temperature} onChange={handleTempSliderChange}/>
                    </InputGroup>
                </Form.Group>
                <Form.Group>
                    <InputGroup className="mb-3 d-flex justify-content-between align-items-center">
                    <Form.Label>Max Completion Tokens &emsp;</Form.Label>
                    <Form.Control type="number" value={tokens} onChange={handleTokenInputChange} min={0} max={1} size='sm' />
                    <Form.Range min={tokens_min} max={tokens_max} step={1} value={tokens} onChange={handleTokenSliderChange}/>
                    </InputGroup>
                </Form.Group>
                <Form.Group>
                    <InputGroup className="mb-3 d-flex justify-content-between align-items-center">
                    <Form.Label>Top P &emsp;</Form.Label>
                    <Form.Control type="number" value={topP} onChange={handleTopPInputChange} min={0} max={1} size='sm' />
                    <Form.Range min={0} max={1} step={0.01} value={topP} onChange={handleTopPSliderChange}/>
                    </InputGroup>
                </Form.Group>
                <Form.Group >
                    <Form.Label>Stop Sequence</Form.Label>
                    <Form.Control type='text' value={stopSequence} onChange={(e) => setStopSequence(e.target.value)} placeholder='Enter the Stop Sequence'></Form.Control>
                </Form.Group>
                <br/>
                <Form.Group className="d-flex align-items-center">
                    <Form.Label>Stream &emsp;</Form.Label>
                    <Form.Check type="switch" id="custom-switch" checked={Stream} onChange={() => setStream(!Stream)}  />
                </Form.Group>
            </Form>
            </div>
            
        </ListGroup.Item>
        <ListGroup.Item as="li" className="d-flex justify-content-between align-items-start" > 
            <div className="ms-2 me-auto">
                <div className="fw-bold"><h6><b>Data Configuration (Optional)</b></h6></div>
                <Form.Group controlId="formFileMultiple" className="mb-3">
                    <Form.Label>Upload your dataset</Form.Label>
                    <Form.Control 
                        type="file" 
                        multiple 
                        accept=".json,application/json" 
                        onChange={handleFileChange} 
                    />
                    {fileUploadErrorMessage && <div className="text-danger">{fileUploadErrorMessage}</div>}
                </Form.Group>
        
            </div>
            <Badge bg="secondary" pill> <FaInfo/> </Badge>
        </ListGroup.Item>
        <ListGroup.Item as="li" className="d-flex justify-content-between align-items-start" >
            <div className="ms-2 me-auto">
            <div className="fw-bold"><h6><b>RAG Configuration</b></h6></div>
            <div>Type</div>
            <RAGSelector selectedType={ragType} setSelectedType={setRagType} />
            <Form.Label>Chunk Size [500-5000]&emsp;</Form.Label>
            <Form.Control type="number" value={chunkSize} onChange={(e) => setChunkSize(Number(e.target.value))} min={500} max={5000} defaultValue={1000}  />
            <Form.Label>Chunk Overlap [0-500]&emsp;</Form.Label>
            <Form.Control type="number" value={chunkOverlap} onChange={(e) => setChunkOverlap(Number(e.target.value))} min={0} max={500} defaultValue={100}  />
            <Form.Label>Top K [1-5]&emsp;</Form.Label>
            <Form.Control type="number" value={topK} onChange={(e) => setTopK(Number(e.target.value))} min={0} max={5} defaultValue={3}  />
            
            </div>
            <Badge bg="secondary" pill> <FaInfo/> </Badge>
        </ListGroup.Item>
        <Button variant="outline-primary" onClick={loadConfiguration}>Load Configuration</Button>
    </ListGroup>
    </>
  );

  return (
    <Container fluid >
      <Row className="h-100">
        {/* Sidebar for desktop */}
        <Col md={3} className="d-none d-md-block border-end p-3">
          <h4>Configurator</h4>
          {sidebarContent}
        </Col>

        {/* Main content */}
        <Col md={9} className="h-100 p-4">
        <Row className='h-20'>
            <div className='d-flex justify-content-between align-items-start'>
            <AccessTypeSelector selectedType={accessType} setSelectedType={setAccessType} />
            <ModelSelector selectedModel={model} setSelectedModel={setModel} />
            </div>
        </Row>
          <div className="d-md-none mb-3">
            <Button variant="outline-primary" onClick={() => setShowOffcanvas(true)}> Configurator </Button>
          </div>

          <Card className="h-100">
            <Card.Header>
              <div className="d-flex justify-content-between align-items-center">
                <span>Model: 3-3-70b-versatile
                    &emsp; stuff
                </span>
                <Button variant="link">View code</Button>
              </div>
            </Card.Header>
            
            <Card.Body className="d-flex flex-column align-items-center justify-content-center">
              <div className="text-center">
                <h4>Welcome to the Playground</h4>
                <ul className="text-start">
                  <li>You can start by typing a message</li>
                  <li>Click submit to get a response</li>
                  <li>Use the icon to view the code</li>
                </ul>
              </div>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {/* Offcanvas for mobile */}
      <Offcanvas show={showOffcanvas} onHide={() => setShowOffcanvas(false)} placement="end">
        <Offcanvas.Header closeButton>
          <Offcanvas.Title>Configurator</Offcanvas.Title>
        </Offcanvas.Header>
        <Offcanvas.Body>
          {sidebarContent}
        </Offcanvas.Body>
      </Offcanvas>
    </Container>
  );
};

export default Playground;