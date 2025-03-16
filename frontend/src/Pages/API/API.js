import React, { useState } from 'react';
import { Button, Offcanvas, Container, Row, Col, ListGroup, Card, Form, InputGroup, Table, Spinner, Modal, Badge } from 'react-bootstrap';
import { FaInfo } from 'react-icons/fa';

// Backend linking setup
import axios from 'axios';
import API_URL from '../../config';

// Modules for Configurator
import AccessTypeSelector from '../../Components/API/AccessSelector';
import ModelSelector from '../../Components/API/ModelSelector';
import RAGSelector from '../../Components/API/RAGSelector';

const Playground = () => {
  const [showOffcanvas, setShowOffcanvas] = useState(false);
  const [model, setModel] = useState('llama3-8b-8192');
  const [accessType, setAccessType] = useState('GROQ');
  const [apikey, setAPIKey] = useState('');
  const [stopSequence, setStopSequence] = useState(null);
  const [Stream, setStream] = useState(false);
  const [temperature, setTemperature] = useState(0);
  const [topP, setTopP] = useState(0);
  const [ragType, setRagType] = useState('Vanilla RAG');
  const [chunkSize, setChunkSize] = useState(1000);
  const [chunkOverlap, setChunkOverlap] = useState(100);
  const [topK, setTopK] = useState(3);
  const [fileUploadErrorMessage, setFileUploadErrorMessage] = useState('');
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [duration,setDuration] = useState(0);
  const [submittedConfig, setSubmittedConfig] = useState({
    accessType: '',
    model: '',
    ragType: '',
  });

  const [validated, setValidated] = useState(false);

  var [data,setData] = useState({});
  var [report,setReport] = useState({});
  const formData = new FormData();


  const handleSubmit = (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    
    if (!form.checkValidity()) {
      event.stopPropagation();
      setValidated(true);
      return;
    }else{
      loadConfiguration();
    }
  }

  const tokens_min = 0;
  const tokens_max = 2048;
  const [tokens, setTokens] = useState((tokens_min + tokens_max) / 2);
  
    const validateMandatoryFields = () => {
      if (!model) {
        alert('Please select a model');
        return false;
      }
      if (!accessType) {
        alert('Please select an access type');
        return false;
      }
      if (!ragType) {
        alert('Please select a RAG type');
        return false;
      }
      return true;
    };
  
    const handleTokenInputChange = (e) => {
      let val = e.target.value;
      if (val === "") {
        setTokens("");
        return;
      }
      val = Number(val);
      if (!isNaN(val) && val >= tokens_min && val <= tokens_max) {
        setTokens(val);
      }
    };
  
    const loadConfiguration = async () => {

      if (!validateMandatoryFields()) return;

      const configData ={
        "MODEL_CONFIG": {
            "type": accessType,
            "model": model,
            "api_key": apikey,
            "temperature": temperature,
            "max_tokens": tokens,
            "top_p": topP,
            "stop": stopSequence,
            "stream": Stream
        },
        "RAG_CONFIG": {
            "type": ragType,
            "chunk_size": chunkSize,
            "chunk_overlap": chunkOverlap,
            "top_k": topK
        }
    }
  
      const configWithFiles = {
        ...configData,
        ...(uploadedFiles.length > 0 && {
          FILES: uploadedFiles.map(file => ({
            name: file.name,
            type: file.type,
            size: file.size
          }))
        })
      };
      formData.append('config', JSON.stringify(configWithFiles));
      // Add actual files if present
      uploadedFiles.forEach(file => {
        formData.append('files', file);
      });

      try {
        // Send configuration to backend

        const startTime = Date.now();

        setShowLoading(true);

        const response = await axios.post(API_URL+'benchmarks', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });
  
        console.log('Server response:', response.data);
        setData(response.data['results'])

        setReport(response.data['report'])


        setSubmittedConfig({
          accessType: accessType,
          model: model,
          ragType: ragType
        });

        const endTime = Date.now();
        const time = (endTime - startTime)/ 1000;
        setDuration(time)
        if(response.status === 200){
          setShowLoading(false);
          setCardsVisible(true);
        }else if (response.status === 205){
          setShowLoading(false);
          alert("Unauthorized search")
      }
      } catch (error) {
        setShowLoading(false);
        console.error('Configuration error:', error);
        alert(`Error: ${error.response?.data?.message || error.message}`);
      }
    };
  

  const handleTempSliderChange = (e) => {
    setTemperature(e.target.value);
  };
  const handleTemperatureInputChange = (e) => {
    let val = e.target.value;
    if (val === null) {
      setTemperature(0);
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


  const handleTopPSliderChange = (e) => {
    setTopP(e.target.value);
  };
  const handleTopPInputChange = (e) => {
    let val = e.target.value;
    if (val === null) {
      setTopP(0);
      return;
    }
    val = Number(val);
    if (!isNaN(val) && val >= 0 && val <= 1) {
      setTopP(val);
    }
  };

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
          setUploadedFiles(Array.from(files));  // Store valid files
      } else {
          // Clear the file input if validation fails
          event.target.value = '';
          setUploadedFiles([]); 
      }
    };

      const downloadResults = () => {
        // Convert the response data to a JSON string
        const jsonString = JSON.stringify(data, null, 2);
    
        // Create a Blob from the JSON string
        const blob = new Blob([jsonString], { type: 'application/json' });
    
        // Create a link element
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
    
        // Set the download attribute with the desired file name
        link.download = `${accessType}_${model}_${ragType}.json`;
    
        // Append the link to the body (required for Firefox)
        document.body.appendChild(link);
    
        // Programmatically click the link to trigger the download
        link.click();
    
        // Remove the link from the document
        document.body.removeChild(link);
      };


      const downloadReport = () => {
        // Convert the response data to a JSON string
        const jsonString = JSON.stringify(report, null, 2);
    
        // Create a Blob from the JSON string
        const blob = new Blob([jsonString], { type: 'application/json' });
    
        // Create a link element
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
    
        // Set the download attribute with the desired file name
        link.download = `${accessType}_${model}_${ragType}_report.json`;
    
        // Append the link to the body (required for Firefox)
        document.body.appendChild(link);
    
        // Programmatically click the link to trigger the download
        link.click();
    
        // Remove the link from the document
        document.body.removeChild(link);
      };

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
    const [cardsVisible, setCardsVisible] = useState(false);


    const [AccessShow, setAccessShow] = useState(false);
    const [ModelShow, setModelShow] = useState(false);
    const [DataShow, setDataShow] = useState(false);
    const [RAGShow, setRAGShow] = useState(false);


    const sampleDataset = [
      {
          "Question": "This is a sample",
          "Context": "This is the context related to the question.",
          "Response": "This is the ground truth answer"
      },
      {
          "Question": "What is the hottest planet in our solar system?",
          "Context": "The planets in our solar system vary in temperature due to their distance from the Sun, atmospheric composition, and other factors.",
          "Response": "Venus is the hottest planet in our solar system, with surface temperatures reaching up to 462°C (864°F), due to its thick atmosphere and runaway greenhouse effect."
      }
  ]
  
    const sidebarContent = (
      <>
        <Form noValidate validated={validated} onSubmit={handleSubmit}>
          <ListGroup as="ol" variant='flush'>
            {/* Access Configuration Section */}
            <ListGroup.Item as='li'>
              <div className="fw-bold"><h6><b>Access Configuration &emsp; <Badge bg="secondary" pill onClick={() => setAccessShow(true)} style={{ cursor: 'pointer' }}><FaInfo/></Badge></b></h6></div>
              <Modal show={AccessShow} onHide={() => setAccessShow(false)}>
                <Modal.Header closeButton>
                  <Modal.Title>Access Configuration Info</Modal.Title>
                </Modal.Header>
                <Modal.Body>
                  {/* Add your configuration information here */}
                  <p>This section allows you to select access control, model selection and the API key needed to acccess the models for the evaluator.</p>
                  <ul>
                    <li>Choose control type (Eg. GROQ )</li>
                    <li>Choose the model you want (Eg. llama3-8b-8192)</li>
                    <li>Enter your API key for the access type selected. Here's how you can create an API key for each of the access types - 
                      <ul>
                        <li>For GROQ, click <a href='https://console.groq.com/keys' target="_blank" rel="noopener noreferrer">here</a></li>
                        <li>For OpenAI, click <a href='https://platform.openai.com/api-keys' target="_blank" rel="noopener noreferrer">here</a></li>
                      </ul>
                    </li>
                  </ul>
                </Modal.Body>
                <Modal.Footer>
                  <button className="btn btn-secondary" onClick={() => setAccessShow(false)}> Close </button>
                </Modal.Footer>
              </Modal>
              <div className='d-flex justify-content-between align-items-start'>
                <AccessTypeSelector selectedType={accessType} setSelectedType={setAccessType} />
                <ModelSelector selectedModel={model} setSelectedModel={setModel} accessType={accessType} />
              </div>
              <Form.Floating className="mb-3">
                  <Form.Control required type='password' value={apikey} onChange={(e) => setAPIKey(e.target.value)} placeholder='Enter your API key'/>
                  <label>API Key *</label>
                  <Form.Control.Feedback type="invalid">
                    Please provide an API key
                  </Form.Control.Feedback>
                </Form.Floating>
            </ListGroup.Item>
    
            {/* Model Configuration Section */}
            <ListGroup.Item as="li"> 
              <div>
                <div className="fw-bold"><h6><b>Model Configuration &emsp; <Badge bg="secondary" pill  onClick={() => setModelShow(true)} style={{ cursor: 'pointer' }}><FaInfo/></Badge></b></h6></div>
                <Modal show={ModelShow} onHide={() => setModelShow(false)}>
                  <Modal.Header closeButton>
                    <Modal.Title>Model Configuration Info</Modal.Title>
                  </Modal.Header>
                  <Modal.Body>
                    {/* Add your configuration information here */}
                    <p>This section allows you to configure the hyperparameters of the model you selected. </p>
                    <ul>
                      <li>Temperature [0-1] : </li>
                      <li>Top P [0-1]</li>
                      <li>Stop Sequence</li>
                      <li>Stream</li>
                    </ul>
                  </Modal.Body>
                  <Modal.Footer>
                    <button className="btn btn-secondary" onClick={() => setModelShow(false)}> Close </button>
                  </Modal.Footer>
                </Modal>
                {/* Token Controls */}
                {/* <Form.Group>
                    <InputGroup className="mb-3 d-flex justify-content-between align-items-center">
                    <Form.Label>Max Completion Tokens&emsp;</Form.Label>
                    <Form.Control required type="number" value={tokens} onChange={handleTokenInputChange} min={tokens_min} max={tokens_max} size='sm' />
                    <Form.Range min={tokens_min} max={tokens_max} step={1} value={tokens} onChange={handleTokenSliderChange}/>
                    </InputGroup>
                </Form.Group> */}
                <Form.Group>
                    <InputGroup className="mb-3 d-flex justify-content-between align-items-center">
                    <Form.Label>Temperature &emsp;</Form.Label>
                    <Form.Control required type="number" value={temperature} onChange={handleTemperatureInputChange} min={0} max={1} step={0.01} size='sm' />
                    <Form.Range min={0} max={1} step={0.01} value={temperature} onChange={handleTempSliderChange}/>
                    </InputGroup>
                </Form.Group>
                <Form.Group>
                    <InputGroup className="mb-3 d-flex justify-content-between align-items-center">
                    <Form.Label>Top P &emsp;</Form.Label>
                    <Form.Control required type="number" value={topP} onChange={handleTopPInputChange} min={0} max={1} step={0.01} size='sm' />
                    <Form.Range min={0} max={1} step={0.01} value={topP} onChange={handleTopPSliderChange}/>
                    </InputGroup>
                </Form.Group>
                <Form.Floating className="mb-3">
                  <Form.Control type='text' value={stopSequence} onChange={(e) => setStopSequence(e.target.value)} placeholder='Enter the Stop Sequence'/>
                  <label htmlFor="floatingInputCustom">Stop Sequence</label>
                </Form.Floating>
                <Form.Group className="d-flex align-items-center">
                    <Form.Label>Stream &emsp;</Form.Label>
                    <Form.Check type="switch" id="custom-switch" checked={Stream} onChange={() => setStream(!Stream)}  />
                </Form.Group>
            </div>
        </ListGroup.Item>


        <ListGroup.Item as="li" className="d-flex align-items-start" > 
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
            <Badge bg="secondary" pill onClick={() => setDataShow(true)} style={{ cursor: 'pointer' }} > <FaInfo/> </Badge>
            <Modal show={DataShow} onHide={() => setDataShow(false)}>
              <Modal.Header closeButton>
                <Modal.Title>Data Configuration Info</Modal.Title>
              </Modal.Header>
              <Modal.Body>
                {/* Add your configuration information here */}
                <p>This section allows you to upload your own dataset to evaluate the set configurations. <br/> Please ensure that the dataset is of the following format - </p>
                <pre>
                  <code>
                    {JSON.stringify(sampleDataset, null, 2)}
                  </code>
                </pre>
              </Modal.Body>
              <Modal.Footer>
                <button className="btn btn-secondary" onClick={() => setDataShow(false)}> Close </button>
              </Modal.Footer>
            </Modal>
        </ListGroup.Item>


        <ListGroup.Item as="li" className="d-flex  align-items-start" >
            <div className="ms-2 me-auto">
            <div className="fw-bold"><h6><b>RAG Configuration</b></h6></div>
            <div>Type</div>
            <RAGSelector selectedType={ragType} setSelectedType={setRagType} />
            <br/>
              <p>
              <Form.Floating >
                <Form.Control required style={{width:"180px"}} type="number" value={chunkSize} onChange={(e) => setChunkSize(Number(e.target.value))} min={500} max={5000} step={100} defaultValue={1000}  />
                <label htmlFor="floatingInputCustom">Chunk Size [500-5000]</label>
              </Form.Floating>
              </p>
              <p>
              <Form.Floating >
                <Form.Control  required type="number" value={chunkOverlap} onChange={(e) => setChunkOverlap(Number(e.target.value))} min={0} max={500} step={50} defaultValue={100}   />
                <label htmlFor="floatingInputCustom">Chunk Overlap [0-500]</label>
              </Form.Floating>
              </p>
              <Form.Floating >
                <Form.Control  required type="number" value={topK} onChange={(e) => setTopK(Number(e.target.value))} min={1} max={5} defaultValue={3}    />
                <label htmlFor="floatingInputCustom">Top K [1-5]</label>
              </Form.Floating>
            </div>
            <Badge bg="secondary" pill onClick={() => setRAGShow(true)} style={{ cursor: 'pointer' }} > <FaInfo/> </Badge>
            <Modal show={RAGShow} onHide={() => setRAGShow(false)}>
              <Modal.Header closeButton>
                <Modal.Title>RAG Configuration Info</Modal.Title>
              </Modal.Header>
              <Modal.Body>
                {/* Add your configuration information here */}
                <p>This section allows you to toggle between different types of RAG techniques and adjust some of their common parameters</p>
                <ul>
                  <li>Choose between <a href='https://arxiv.org/pdf/2005.11401' target="_blank" rel="noopener noreferrer">Vanilla RAG</a> [default], <a href='https://arxiv.org/pdf/2401.18059' target="_blank" rel="noopener noreferrer">RAPTOR</a> or <a href='https://arxiv.org/pdf/2404.16130' target="_blank" rel="noopener noreferrer">Graph RAG</a> for evaluations</li>
                  <li>Chunk Size</li>
                  <li>Chunk Overlap</li>
                  <li>Top K</li>
                </ul>
              </Modal.Body>
              <Modal.Footer>
                <button className="btn btn-secondary" onClick={() => setRAGShow(false)}> Close </button>
              </Modal.Footer>
            </Modal>
        </ListGroup.Item>
        <br/>
        <Button variant="outline-primary" type="submit">Load Configuration</Button>
    </ListGroup>
    </Form>
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
          <div className="d-md-none mb-3">
            <Button variant="outline-primary" onClick={() => setShowOffcanvas(true)}> Configurator </Button>
          </div>

          <Card style={{ maxWidth: "100%" }}>
            <Card.Header>
              <div className="d-flex justify-content-between align-items-center">
                <span><b>Config</b>: {submittedConfig.accessType || 'GROQ'} | <b>Model</b>: {submittedConfig.model || 'llama3-8b-8192'} | <b>RAG</b> : {submittedConfig.ragType || 'Vanilla RAG'} </span>
                <span><b>Time Taken</b>: {duration}s</span>
              </div>
            </Card.Header>
            <Card.Body>
              <div style={{ overflowX: 'auto' }}>
                <Table striped bordered hover style={{ 
                  minWidth: `${(1 + Object.keys(data).length) > 6 ? (1 + Object.keys(data).length) * 150 : 'auto'}px`
                }}>
                  <thead>
                    <tr>
                      <th>Metric / Dataset</th>
                      {Object.keys(data).map((dataset) => (
                        <th key={dataset}>{dataset.replace(/_/g, ' ').toUpperCase()}</th>
                      ))}
                    </tr>
                  </thead>
                  {showLoading ? (
                    <div style={loadingScreenStyle} className="loading-screen">
                      <Spinner animation="border" role="status"></Spinner>
                    </div>
                  ) : cardsVisible ? (
                    <tbody>
                      {[
                        { label: 'Cosine Similarity', path: 'cosine_similarity' },
                        { label: 'BLEU', path: 'bleu' },
                        { label: 'ROUGE-1', path: 'rouge.rouge1' },
                        { label: 'ROUGE-2', path: 'rouge.rouge2' },
                        { label: 'ROUGE-L', path: 'rouge.rougeL' },
                        { label: 'ROUGE-Lsum', path: 'rouge.rougeLsum' },
                        { label: 'METEOR', path: 'meteor' },
                        { label: 'Roberta-Entailment', path: 'roberta_nli.entailment' },
                        { label: 'Roberta-Contradiction', path: 'roberta_nli.contradiction' },
                        { label: 'Roberta-Neutral', path: 'roberta_nli.neutral' },
                      ].map((metric) => {
                        const getValue = (dataset, type) => {
                          const parts = metric.path.split('.');
                          let value = data[dataset][type];
                          parts.forEach((part) => {
                            value = value[part];
                          });
                          return value;
                        };

                        return (
                          <tr key={metric.path}>
                            <td><strong>{metric.label}</strong></td>
                            {Object.keys(data).map((dataset) => {
                              const mean = getValue(dataset, 'mean');
                              const variance = getValue(dataset, 'variance');
                              return (
                                <td key={`${dataset}-${metric.path}`}>
                                  {mean.toFixed(3)} ± {variance.toFixed(3)}
                                </td>
                              );
                            })}
                          </tr>
                        );
                      })}
                    </tbody>
                  ) : null}
                </Table>
              </div>
            </Card.Body>
            <Card.Header>
              <div className="d-flex justify-content-between align-items-center">
                <Button variant="link" onClick={downloadResults}>Download Results</Button>
                <Button variant="link" onClick={downloadReport}>Download Comprehensive Report</Button>
                <Button variant="link" target="_blank" href='https://github.com/Ritvik-G/in-situ_eval' rel="noopener noreferrer">View code</Button>
              </div>
            </Card.Header>
          </Card>
          <br/>
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