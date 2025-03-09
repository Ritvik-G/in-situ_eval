import React, { useState } from 'react';
import { 
  Button, Offcanvas, Container, Row, Col, ListGroup, Card, Form,
  InputGroup, Table, Spinner, Modal, Badge 
} from 'react-bootstrap';
import { FaInfo } from 'react-icons/fa';
import axios from 'axios';
import API_URL from '../../config';
import AccessTypeSelector from '../../Components/API/AccessSelector';
import ModelSelector from '../../Components/API/ModelSelector';
import RAGSelector from '../../Components/API/RAGSelector';

const Playground = () => {
  // State Management
  const [uiState, setUiState] = useState({
    showOffcanvas: false,
    showLoading: false,
    cardsVisible: false,
    validated: false,
    modals: { access: false, model: false, data: false, rag: false }
  });

  const [config, setConfig] = useState({
    model: 'llama3-8b-8192',
    accessType: 'GROQ',
    apikey: '',
    ragType: 'Vanilla RAG',
    stopSequence: null,
    stream: false,
    temperature: 0,
    topP: 0,
    tokens: 1024,
    chunkSize: 1000,
    chunkOverlap: 100,
    topK: 3
  });

  const [fileState, setFileState] = useState({
    uploadError: '',
    uploadedFiles: [],
    duration: 0
  });

  const [results, setResults] = useState({
    data: {},
    report: {},
    submittedConfig: { accessType: '', model: '', ragType: '' }
  });

  // Constants
  const PARAM_LIMITS = {
    tokens: { min: 0, max: 2048 },
    temperature: { min: 0, max: 1, step: 0.01 },
    topP: { min: 0, max: 1, step: 0.01 },
    chunkSize: { min: 500, max: 5000, step: 100 },
    chunkOverlap: { min: 0, max: 500, step: 50 },
    topK: { min: 1, max: 5 }
  };

  // Handlers
  const handleStateUpdate = (category, updates) => {
    if (category === 'ui') {
      setUiState(prev => ({ ...prev, ...updates }));
    } else if (category === 'config') {
      setConfig(prev => ({ ...prev, ...updates }));
    } else if (category === 'file') {
      setFileState(prev => ({ ...prev, ...updates }));
    } else if (category === 'results') {
      setResults(prev => ({ ...prev, ...updates }));
    }
  };

  const handleParamChange = (param, value) => {
    const limits = PARAM_LIMITS[param];
    const numericValue = Number(value);
    
    if (!isNaN(numericValue) && numericValue >= limits.min && numericValue <= limits.max) {
      handleStateUpdate('config', { [param]: numericValue });
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    const form = event.currentTarget;

    if (!form.checkValidity()) {
      event.stopPropagation();
      handleStateUpdate('ui', { validated: true });
      return;
    }

    if (!validateMandatoryFields()) return;

    const formData = new FormData();
    const configData = {
      MODEL_CONFIG: {
        type: config.accessType,
        model: config.model,
        api_key: config.apikey,
        temperature: config.temperature,
        max_tokens: config.tokens,
        top_p: config.topP,
        stop: config.stopSequence,
        stream: config.stream
      },
      RAG_CONFIG: {
        type: config.ragType,
        chunk_size: config.chunkSize,
        chunk_overlap: config.chunkOverlap,
        top_k: config.topK
      }
    };

    formData.append('config', JSON.stringify({
      ...configData,
      ...(fileState.uploadedFiles.length > 0 && {
        FILES: fileState.uploadedFiles.map(file => ({
          name: file.name,
          type: file.type,
          size: file.size
        }))
      })
    }));

    fileState.uploadedFiles.forEach(file => formData.append('files', file));

    try {
      const startTime = Date.now();
      handleStateUpdate('ui', { showLoading: true });

      const response = await axios.post(`${API_URL}benchmarks`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      handleStateUpdate('results', {
        data: response.data.results,
        report: response.data.report,
        submittedConfig: {
          accessType: config.accessType,
          model: config.model,
          ragType: config.ragType
        }
      });

      handleStateUpdate('file', { duration: (Date.now() - startTime) / 1000 });
      handleStateUpdate('ui', { cardsVisible: true });
    } catch (error) {
      console.error('Configuration error:', error);
      alert(`Error: ${error.response?.data?.message || error.message}`);
    } finally {
      handleStateUpdate('ui', { showLoading: false });
    }
  };

  // Helper Functions
  const validateMandatoryFields = () => {
    const requiredFields = [
      [!config.model, 'Please select a model'],
      [!config.accessType, 'Please select an access type'],
      [!config.ragType, 'Please select a RAG type']
    ];

    for (const [condition, message] of requiredFields) {
      if (condition) {
        alert(message);
        return false;
      }
    }
    return true;
  };

  const handleFileChange = (event) => {
    const files = Array.from(event.target.files);
    const isValid = files.every(file => 
      file.type === 'application/json' && 
      files.reduce((sum, f) => sum + f.size, 0) <= 20 * 1024 * 1024
    );

    if (!isValid) {
      handleStateUpdate('file', {
        uploadError: files.some(f => f.type !== 'application/json') 
          ? 'Only JSON files are allowed' 
          : 'Total size exceeds 20MB'
      });
      event.target.value = '';
      return;
    }

    handleStateUpdate('file', {
      uploadError: '',
      uploadedFiles: files
    });
  };

  const downloadFile = (content, suffix) => {
    const blob = new Blob([JSON.stringify(content, null, 2)], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `${config.accessType}_${config.model}_${config.ragType}_${suffix}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // UI Components
  const ConfigModal = ({ show, onHide, title, children }) => (
    <Modal show={show} onHide={onHide}>
      <Modal.Header closeButton>
        <Modal.Title>{title}</Modal.Title>
      </Modal.Header>
      <Modal.Body>{children}</Modal.Body>
      <Modal.Footer>
        <Button variant="secondary" onClick={onHide}>Close</Button>
      </Modal.Footer>
    </Modal>
  );

  const ParamControl = ({ label, param, limits }) => (
    <Form.Group className="mb-3">
      <InputGroup className="d-flex justify-content-between align-items-center">
        <Form.Label>{label}</Form.Label>
        <Form.Control
          type="number"
          value={config[param]}
          onChange={(e) => handleParamChange(param, e.target.value)}
          min={limits.min}
          max={limits.max}
          step={limits.step || 1}
          size="sm"
        />
      </InputGroup>
      <Form.Range
        min={limits.min}
        max={limits.max}
        step={limits.step || 1}
        value={config[param]}
        onChange={(e) => handleParamChange(param, e.target.value)}
      />
    </Form.Group>
  );

  const LoadingScreen = () => (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      background: 'rgba(255, 255, 255, 0.7)',
      zIndex: 1000
    }}>
      <Spinner animation="border" role="status">
        <span className="visually-hidden">Loading...</span>
      </Spinner>
    </div>
  );

  // Main Render
  return (
    <Container fluid>
      <Row className="h-100">
        {/* Sidebar */}
        <Col md={3} className="d-none d-md-block border-end p-3">
          <h4>Configurator</h4>
          <Form noValidate validated={uiState.validated} onSubmit={handleSubmit}>
            <ListGroup variant="flush">
              
              {/* Access Configuration */}
              <ListGroup.Item>
                <div className="fw-bold d-flex justify-content-between align-items-center">
                  <h6><b>Access Configuration</b></h6>
                  <Badge bg="secondary" pill onClick={() => handleStateUpdate('ui', { modals: { ...uiState.modals, access: true } })}>
                    <FaInfo />
                  </Badge>
                </div>
                <div className="d-flex justify-content-between gap-2 mt-2">
                  <AccessTypeSelector
                    selectedType={config.accessType}
                    setSelectedType={(type) => handleStateUpdate('config', { accessType: type })}
                  />
                  <ModelSelector
                    selectedModel={config.model}
                    setSelectedModel={(model) => handleStateUpdate('config', { model })}
                    accessType={config.accessType}
                  />
                </div>
              </ListGroup.Item>

              {/* Model Configuration */}
              <ListGroup.Item>
                <div className="fw-bold d-flex justify-content-between align-items-center">
                  <h6><b>Model Configuration</b></h6>
                  <Badge bg="secondary" pill onClick={() => handleStateUpdate('ui', { modals: { ...uiState.modals, model: true } })}>
                    <FaInfo />
                  </Badge>
                </div>
                <Form.Floating className="mb-3">
                  <Form.Control
                    required
                    type="password"
                    value={config.apikey}
                    onChange={(e) => handleStateUpdate('config', { apikey: e.target.value })}
                    placeholder="API Key"
                  />
                  <label>API Key *</label>
                </Form.Floating>
                <ParamControl
                  label="Max Tokens"
                  param="tokens"
                  limits={PARAM_LIMITS.tokens}
                />
                <ParamControl
                  label="Temperature"
                  param="temperature"
                  limits={PARAM_LIMITS.temperature}
                />
                <ParamControl
                  label="Top P"
                  param="topP"
                  limits={PARAM_LIMITS.topP}
                />
                <Form.Floating className="mb-3">
                  <Form.Control
                    type="text"
                    value={config.stopSequence || ''}
                    onChange={(e) => handleStateUpdate('config', { stopSequence: e.target.value })}
                    placeholder="Stop Sequence"
                  />
                  <label>Stop Sequence</label>
                </Form.Floating>
                <Form.Check
                  type="switch"
                  label="Stream"
                  checked={config.stream}
                  onChange={() => handleStateUpdate('config', { stream: !config.stream })}
                />
              </ListGroup.Item>

              {/* Data Configuration */}
              <ListGroup.Item>
                <div className="fw-bold d-flex justify-content-between align-items-center">
                  <h6><b>Data Configuration (Optional)</b></h6>
                  <Badge bg="secondary" pill onClick={() => handleStateUpdate('ui', { modals: { ...uiState.modals, data: true } })}>
                    <FaInfo />
                  </Badge>
                </div>
                <Form.Group className="mt-2">
                  <Form.Control
                    type="file"
                    multiple
                    accept=".json"
                    onChange={handleFileChange}
                  />
                  {fileState.uploadError && (
                    <div className="text-danger small mt-1">{fileState.uploadError}</div>
                  )}
                </Form.Group>
              </ListGroup.Item>

              {/* RAG Configuration */}
              <ListGroup.Item>
                <div className="fw-bold d-flex justify-content-between align-items-center">
                  <h6><b>RAG Configuration</b></h6>
                  <Badge bg="secondary" pill onClick={() => handleStateUpdate('ui', { modals: { ...uiState.modals, rag: true } })}>
                    <FaInfo />
                  </Badge>
                </div>
                <RAGSelector
                  selectedType={config.ragType}
                  setSelectedType={(type) => handleStateUpdate('config', { ragType: type })}
                />
                <ParamControl
                  label="Chunk Size"
                  param="chunkSize"
                  limits={PARAM_LIMITS.chunkSize}
                />
                <ParamControl
                  label="Chunk Overlap"
                  param="chunkOverlap"
                  limits={PARAM_LIMITS.chunkOverlap}
                />
                <ParamControl
                  label="Top K"
                  param="topK"
                  limits={PARAM_LIMITS.topK}
                />
              </ListGroup.Item>

              <ListGroup.Item>
                <Button variant="outline-primary" type="submit" className="w-100">
                  Load Configuration
                </Button>
              </ListGroup.Item>
            </ListGroup>
          </Form>
        </Col>

        {/* Main Content */}
        <Col md={9} className="p-4">
          <div className="d-md-none mb-3">
            <Button variant="outline-primary" onClick={() => handleStateUpdate('ui', { showOffcanvas: true })}>
              Configurator
            </Button>
          </div>

          <Card>
            <Card.Header className="d-flex justify-content-between">
              <div>
                <b>Config:</b> {results.submittedConfig.accessType} |{' '}
                <b>Model:</b> {results.submittedConfig.model} |{' '}
                <b>RAG:</b> {results.submittedConfig.ragType}
              </div>
              <div><b>Time Taken:</b> {fileState.duration}s</div>
            </Card.Header>

            <Card.Body>
              {uiState.showLoading && <LoadingScreen />}
              <div style={{ overflowX: 'auto' }}>
                <Table striped bordered hover>
                  <thead>
                    <tr>
                      <th>Metric/Dataset</th>
                      {Object.keys(results.data).map(dataset => (
                        <th key={dataset}>{dataset.replace(/_/g, ' ').toUpperCase()}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      'cosine_similarity', 'bleu', 
                      'rouge.rouge1', 'rouge.rouge2', 
                      'rouge.rougeL', 'rouge.rougeLsum',
                      'meteor', 'roberta_nli.entailment',
                      'roberta_nli.contradiction', 'roberta_nli.neutral'
                    ].map(metricPath => {
                      const [mainMetric, subMetric] = metricPath.split('.');
                      return (
                        <tr key={metricPath}>
                          <td><b>{metricPath.replace('.', '-').replace(/_/g, ' ').toUpperCase()}</b></td>
                          {Object.keys(results.data).map(dataset => {
                            const value = subMetric 
                              ? results.data[dataset].mean[mainMetric][subMetric]
                              : results.data[dataset].mean[mainMetric];
                            const variance = subMetric
                              ? results.data[dataset].variance[mainMetric][subMetric]
                              : results.data[dataset].variance[mainMetric];
                            return (
                              <td key={`${dataset}-${metricPath}`}>
                                {value?.toFixed(3) || 'N/A'} ± {variance?.toFixed(3) || 'N/A'}
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </Table>
              </div>
            </Card.Body>

            <Card.Header className="d-flex justify-content-between">
              <Button variant="link" onClick={() => downloadFile(results.data, 'results')}>
                Download Results
              </Button>
              <Button variant="link" onClick={() => downloadFile(results.report, 'report')}>
                Download Report
              </Button>
            </Card.Header>
          </Card>
        </Col>
      </Row>

      {/* Modals */}
      <ConfigModal
        show={uiState.modals.access}
        onHide={() => handleStateUpdate('ui', { modals: { ...uiState.modals, access: false } })}
        title="Access Configuration Info"
      >
        <p>Configure API access parameters including provider selection and authentication.</p>
      </ConfigModal>

      <ConfigModal
        show={uiState.modals.model}
        onHide={() => handleStateUpdate('ui', { modals: { ...uiState.modals, model: false } })}
        title="Model Configuration Info"
      >
        <p>Adjust model parameters like temperature and token limits to control generation behavior.</p>
      </ConfigModal>

      <ConfigModal
        show={uiState.modals.data}
        onHide={() => handleStateUpdate('ui', { modals: { ...uiState.modals, data: false } })}
        title="Data Configuration Info"
      >
        <p>Upload custom datasets in JSON format (max 20MB total size).</p>
      </ConfigModal>

      <ConfigModal
        show={uiState.modals.rag}
        onHide={() => handleStateUpdate('ui', { modals: { ...uiState.modals, rag: false } })}
        title="RAG Configuration Info"
      >
        <p>Configure Retrieval-Augmented Generation parameters including chunking and retrieval settings.</p>
      </ConfigModal>

      {/* Mobile Offcanvas */}
      <Offcanvas show={uiState.showOffcanvas} onHide={() => handleStateUpdate('ui', { showOffcanvas: false })} placement="end">
        <Offcanvas.Header closeButton>
          <Offcanvas.Title>Configurator</Offcanvas.Title>
        </Offcanvas.Header>
        <Offcanvas.Body>
          {/* Reuse the sidebar form components */}
          <Form noValidate validated={uiState.validated} onSubmit={handleSubmit}>
            {/* ... (Same form content as desktop sidebar) ... */}
          </Form>
        </Offcanvas.Body>
      </Offcanvas>
    </Container>
  );
};

export default Playground;