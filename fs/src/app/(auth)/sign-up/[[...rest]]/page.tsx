"use client"

import React from "react";
import { SignUp} from "@clerk/nextjs";
import FooterSection from "@/components/sections/FooterSection";

const Page = () => {
  return (
    <>
    <div className="min-h-screen flex flex-1 items-center justify-center bg-gray-50 px-4 sm:px-6 lg:px-8 mt-14">
        <SignUp />
    </div>

    <FooterSection/>
    </>
  );
};

export default Page;
