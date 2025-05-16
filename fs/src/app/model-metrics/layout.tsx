import CarouselNavbar from '@/components/sections/CarouselNavbar'

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <>
      <CarouselNavbar />
      {children}
    </>
  );
}
