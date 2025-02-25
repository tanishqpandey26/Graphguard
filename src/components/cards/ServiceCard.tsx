import React from "react";
import MainButton from "../common/MainButton";

interface IProps {
  iconUrl: string;
  title: string;
  description: string;
  cardClass?: string; 
  iconClass?: string; 
  titleClass?: string; 
  descriptionClass?: string;
  action?: () => void;
}

function ServiceCard({
  iconUrl,
  title,
  description,
  cardClass = "", 
  iconClass = "",
  titleClass = "",
  descriptionClass = "",
  action,
}: IProps) {
  return (
    <div
      className={`flex flex-grow flex-col gap-[2.56rem] pt-[1.91rem] pb-[2.81rem] px-[2.56rem] items-center service-card-shadow rounded-[1.75rem] ${cardClass}`}
    >
      <div>
        <img src={iconUrl} alt="service icon" className={`service-card-icon ${iconClass}`} />
      </div>
      <p className={`text-[2.25rem] font-[700] ${titleClass}`}>{title}</p>
      <p className={`text-normal ${descriptionClass}`}>{description}</p>
      <MainButton text="Learn More" action={action} classes="w-[10.125rem]" />
    </div>
  );
}

export default ServiceCard;