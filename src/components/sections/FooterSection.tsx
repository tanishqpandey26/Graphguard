import React from "react";
import { Separator } from "../ui/separator";
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
              href='https://www.linkedin.com/in/pandey26tanishq'
              aria-label='LinkedIn Profile'
              title='LinkedIn'
            >
              <Linkedin/>
            </a>

            <a
              href='https://github.com/tanishqpandey26'
              aria-label='GitHub Profile'
              title='GitHub'
            >
              <Github/>
            </a>

            <a
              href='https://twitter.com/tanishqvatsa26'
              aria-label='Twitter Profile'
              title='Twitter'
            >
             <Twitter/>
            </a>

            <a
              href='https://digital-resume-fawn.vercel.app/'
              aria-label='View Digital Resume'
              title='Resume'
            >
              <FileText/>
            </a>
          </div>
        </div>

        <div className='bottom'>
          <div>
            <h4>Project</h4>
            <a href='https://tripper-ui.vercel.app/'>Tripper</a>
            <a href='https://cotlog-social-blog-website.vercel.app/'>Cotlog</a>
            <a href='https://digital-resume-fawn.vercel.app'>Digital Resume</a>
          </div>

          <div>
            <h4>Community</h4>
            <a href='https://github.com/tanishqpandey26'>GitHub</a>
            <a href='https://www.linkedin.com/in/pandey26tanishq'>LinkedIn</a>
            <a href='https://twitter.com/tanishqvatsa26'>Twitter</a>
          </div>

          <div>
            <h4>Help</h4>
            <a href='/service'>Support</a>
            <a href='/contact'>Troubleshooting</a>
            <a href='/contact'>Contact Us</a>
          </div>

          <div>
            <h4>Others</h4>
            <a href='/about'>Terms of Service</a>
            <a href='/about'>Privacy Policy</a>
            <a href='/about'>License</a>
          </div>
        </div>

      <div className="footer-copyright">
      <h3> Copyright <Copyright />Tanishq Pandey</h3>
      </div>

      </div>

      <Separator />
    </>
  );
}

export default FooterSection;
