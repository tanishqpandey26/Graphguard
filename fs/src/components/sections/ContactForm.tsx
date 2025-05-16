"use client";

import React, { useState, useRef } from "react";
import "./ContactFormStyles.css";
import { Button } from "../ui/button";

function ContactForm() {
    const [result, setResult] = useState("");
    const formRef = useRef<HTMLFormElement>(null); 

    const onSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        setResult("Sending...");

        const formData = new FormData(event.currentTarget);

        
        const accessKey = process.env.NEXT_PUBLIC_WEB3FORMS_ACCESS_KEY;
        if (!accessKey) {
            setResult("Error: Access Key is missing.");
            return;
        }

        formData.append("access_key", accessKey);

        const response = await fetch("https://api.web3forms.com/submit", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            setResult("Form Submitted Successfully");

            
            if (formRef.current) {
                formRef.current.reset();
            }
        } else {
            console.log("Error", data);
            setResult(data.message);
        }
    };

    return (
        <div className="contact-container">

            <div className="contact-heading">
                <h1>Need help? Contact us!</h1>
                <h2>Send a message to us!</h2>
            </div>
    
            
            <div className="contact-content">
                <div className="contact-image">
                    <img src="/images/contact.png" alt="contact-image" />
                </div>
    
                <div className="form-container"> 
                    <form>
                        <input type="text" name="name" placeholder="Name" required />
                        <input type="email" name="email" placeholder="Email" required />
                        <textarea name="message" placeholder="Message" rows={4} required />
                        <Button type="submit">Send Message</Button>
                    </form>
                </div>
            </div>
        </div>
    );
};

export default ContactForm;
