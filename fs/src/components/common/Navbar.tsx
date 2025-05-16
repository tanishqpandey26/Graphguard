"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { X } from "lucide-react";


function Navbar() {
  const [menu, setMenu] = useState(false);
  const router = useRouter(); 

  const toggleMenu = () => {
    setMenu(!menu);
  };

  const handleNavigation = (path: string) => {
    router.push(path);
    setMenu(false); 
  };

  return (
    <div className="md:sticky md:top-0 md:shadow-none z-20">
      
      <div className="hidden lg:block animate-in fade-in zoom-in bg-white p-4">
        <div className="flex justify-between md:mx-[9rem] items-center">
          <div>
            <h1 className="text-xl font-bold cursor-pointer" onClick={() => handleNavigation("/")}>
              GraphGuard
            </h1>
          </div>
       
          <div className="flex items-center gap-[40px] select-none">

            <button
              onClick={() => handleNavigation("/about")}
              className=" cursor-pointer font-[500] text-indigo-600 hover:bg-slate-100 rounded-md py-2 px-4 transition-all"
            >
              About
            </button>

            <button
              onClick={() => handleNavigation("/team")}
              className=" cursor-pointer font-[500] text-indigo-600 hover:bg-slate-100 rounded-md py-2 px-4 transition-all"
            >
              Team
            </button>

            <button
              onClick={() => handleNavigation("/docs")}
              className=" cursor-pointer font-[500] text-indigo-600 hover:bg-slate-100 rounded-md py-2 px-4 transition-all"
            >
              Docs 
            </button>
            
            <button
              onClick={() => handleNavigation("/model-metrics")}
              className=" cursor-pointer font-[500] text-indigo-600 hover:bg-slate-100 rounded-md py-2 px-4 transition-all"
            >
              Model-Metrics
            </button>
            
            <button
              onClick={() => handleNavigation("/dashboard")}
              className=" cursor-pointer font-[500] text-indigo-600 hover:bg-slate-100 rounded-md py-2 px-4 transition-all"
            >
              Dashboard
            </button>

          </div>
        </div>
      </div>


      <div
        className={`block lg:hidden shadow-sm fixed top-0 w-full z-[999] bg-white py-4 animate-in fade-in zoom-in ${menu ? " bg-primary py-2" : ""}`}
      >
        <div className="flex justify-between mx-[10px]">
          <div className="flex gap-[50px] text-[16px] items-center select-none">
            <h1 className="text-xl font-bold cursor-pointer" onClick={() => handleNavigation("/")}>
              GraphGuard
            </h1>
          </div>
          <div className="flex items-center gap-[40px]">
            {menu ? (
              <X className="cursor-pointer animate-in fade-in zoom-in text-black" onClick={toggleMenu} />
            ) : (
              <img
                src="/svgs/hamburger.svg"
                alt="menu"
                className="cursor-pointer animate-in fade-in zoom-in"
                onClick={toggleMenu}
              />
            )}
          </div>
        </div>
        
        {menu && (
          
          <div className="my-8 select-none animate-in slide-in-from-right">
            <div className="flex flex-col gap-8 mt-8 mx-4">
              
              <button onClick={() => handleNavigation("/about")} className=" cursor-pointer font-[500] text-indigo-600 hover:bg-slate-100 rounded-md py-2 px-4 transition-all">
                About
              </button>

              <button onClick={() => handleNavigation("/team")} className=" cursor-pointer font-[500] text-indigo-600 hover:bg-slate-100 rounded-md py-2 px-4 transition-all">
                Team
              </button>

              <button onClick={() => handleNavigation("/docs")} className=" cursor-pointer font-[500] text-indigo-600 hover:bg-slate-100 rounded-md py-2 px-4 transition-all">
                Docs
              </button>

              <button
              onClick={() => handleNavigation("/model-metrics")}
              className=" cursor-pointer font-[500] text-indigo-600 hover:bg-slate-100 rounded-md py-2 px-4 transition-all"
            >
              Model-Metrics
            </button>

            <button
              onClick={() => handleNavigation("/dashboard")}
              className=" cursor-pointer font-[500] text-indigo-600 hover:bg-slate-100 rounded-md py-2 px-4 transition-all"
            >
              Dashboard
            </button>

            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default Navbar;
