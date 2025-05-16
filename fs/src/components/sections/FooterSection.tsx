import React from "react";
import "./FooterSection.css";
import { Copyright, Linkedin, Github, FileText, Twitter } from "lucide-react";

function FooterSection() {
  return (
    <>
      <div className='footer'>

        <div className='top'>
          <div>
            <h1>GraphGuard</h1>
            <p>Payment fraud detection</p>
          </div>

          <div>
          <a
              href='https://github.com/tanishqpandey26/Graphguard'
              aria-label='GitHub Profile'
              title='GitHub'
            >
              <Github/>
            </a>

            <a
              href='https://www.linkedin.com/in/pandey26tanishq'
              aria-label='LinkedIn Profile'
              title='LinkedIn'
            >
              <Linkedin/>
            </a>

            <a
              href='https://twitter.com/tanishqvatsa26'
              aria-label='Twitter Profile'
              title='Twitter'
            >
             <Twitter/>
            </a>
         </div>
         
        </div>

        <div className='bottom'>

          <div>
            <h4>Service</h4>
            <a href='/'>Home</a>
            <a href='/service'>About</a>
            <a href='/team'>Team</a>
          </div>

          <div>
            <h4>Support</h4>
            <a href='/docs'>Docs</a>
            <a href='/model-metrics'>Model Metrics</a>
            <a href='/about'>Contact Us</a>
          </div>

          <div>
            <h4>Community</h4>
            <a href='https://github.com/tanishqpandey26'>GitHub</a>
            <a href='https://www.linkedin.com/in/pandey26tanishq'>LinkedIn</a>
            <a href='https://twitter.com/tanishqvatsa26'>Twitter</a>
          </div>

          <div>
            <h4>Projects</h4>
            <a href='https://tripper-ui.vercel.app/'>Tripper</a>
            <a href='https://cotlog-social-blog-website.vercel.app/'>Cotlog</a>
            <a href='https://digital-resume-fawn.vercel.app'>Digital Resume</a>
          </div>

        </div>

      <div className="footer-copyright">
      <h3> Copyright <Copyright />GraphGuard</h3>
      </div>

      </div>
    </>
  );
}

export default FooterSection;
