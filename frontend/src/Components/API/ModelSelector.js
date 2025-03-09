import React, { useState } from 'react';
import { Dropdown, Form } from 'react-bootstrap';
import modelsConfig from '../../Data/Models.json'; // Import the JSON file

export default function ModelSelector({ selectedModel, setSelectedModel, accessType }) {
  const [search, setSearch] = useState('');
  const [show, setShow] = useState(false);

  // Get the model groups based on the selected accessType
  const modelGroups = accessType === 'GROQ' ? modelsConfig.GROQ : modelsConfig.OpenAI;

  // Filter model groups based on search input
  const filtered = modelGroups.filter(group => 
    group.provider.toLowerCase().includes(search.toLowerCase()) ||
    group.models.some(model => model.name.toLowerCase().includes(search.toLowerCase()))
  );

  return (
    <div className="mb-3">
      <Dropdown show={show} onToggle={setShow}>
        <Dropdown.Toggle variant="outline-secondary" className="w-100 text-start">
          {selectedModel || "Select a Model"}
        </Dropdown.Toggle>

        <Dropdown.Menu className="p-2" style={{ maxHeight: '300px', overflowY: 'auto' }}>
          <Form.Control
            type="text"
            placeholder="Search..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />

          {filtered.map((group, i) => (
            <div key={i}>
              <Dropdown.Header>{group.provider}</Dropdown.Header>
              {group.models.map((model, j) => (
                <Dropdown.Item key={j} onClick={() => {
                  setSelectedModel(model.name); // Use model.name instead of model
                  setShow(false);
                  setSearch('');
                }}>
                  {model.name} {/* Display model name */}
                </Dropdown.Item>
              ))}
            </div>
          ))}
        </Dropdown.Menu>
      </Dropdown>
    </div>
  );
}