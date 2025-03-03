import NavBar from "@/components/common/NavBar";
import FooterSection from "@/components/sections/FooterSection";
import HeroSection from "@/components/sections/HeroSection";
import ContactForm from "@/components/sections/ContactForm";
import ServiceSection from "@/components/sections/ServiceSection";
import PerformanceSection from "@/components/sections/PerformanceSection";
import VideoPlayerSection from "@/components/sections/VideoPlayerSection";

export default function Home() {
  return (
    <main>
      <NavBar />

      <div className="mt-24 md:32 lg:mt-8 px-4  md:px-[3rem]">
        <HeroSection />
        <ServiceSection />
        <VideoPlayerSection />
      </div>

      <PerformanceSection />
      <ContactForm />
      <FooterSection />
    </main>
  );
}