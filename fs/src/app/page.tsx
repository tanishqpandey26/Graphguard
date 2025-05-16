import Navbar from "@/components/common/Navbar";
import FooterSection from "@/components/sections/FooterSection";
import HeroSection from "@/components/sections/HeroSection";
import ContactForm from "@/components/sections/ContactForm";
import ServiceSection from "@/components/sections/ServiceSection";
import PerformanceSection from "@/components/sections/PerformanceSection";
import VideoPlayerSection from "@/components/sections/VideoPlayerSection";
import DocsSection from "@/components/sections/DocsSection";

export default function Home() {
  return (
    <main>
      <Navbar />
      <div className="mt-5">
      <HeroSection />
      </div>
      <ServiceSection />
      <VideoPlayerSection />
      <PerformanceSection />
      <DocsSection/>
      <ContactForm />
      <FooterSection />
    </main>
  );
}
