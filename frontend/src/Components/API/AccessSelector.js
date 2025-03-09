import React, { useState } from 'react';
import { Dropdown, Form } from 'react-bootstrap';

const accessTypes = ["GROQ", "OpenAI"];

export default function AccessTypeSelector({ selectedType, setSelectedType }) {
  const [show, setShow] = useState(false);
  const [search, setSearch] = useState('');

  const filtered = accessTypes.filter(type => 
    type.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div >
      
        <Dropdown show={show} onToggle={setShow}>
          <Dropdown.Toggle variant="outline-secondary" className="text-start">
            {selectedType}
          </Dropdown.Toggle>

          <Dropdown.Menu className='p-2'>
            <Form.Control
              type="text"
              placeholder="Search..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />

            {filtered.map((type, i) => (
              <Dropdown.Item 
                key={i}
                onClick={() => {
                  setSelectedType(type);
                  setShow(false);
                  setSearch('');
                }}
              >
                {type}
              </Dropdown.Item>
            ))}
          </Dropdown.Menu>
        </Dropdown>
    </div>
  );
}