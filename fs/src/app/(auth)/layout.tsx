import { ReactNode } from "react"
import Navbar from "@/components/common/Navbar";
import FooterSection from "@/components/sections/FooterSection";

const Layout = ({ children }: { children: ReactNode }) => {
  return (
    <>
      <Navbar />
      {children}
    </>
  )
}

export default Layout
